// Build:
//   nvcc -O3 kernel_ilu0_pcg.cu -o kernel_ilu0_pcg -lcusparse
// Run:
//   ./kernel_ilu0_pcg
//
// ILU(0)-preconditioned CG (PCG) -- the preconditioner actually expected
// to reduce iteration count (unlike Jacobi, see kernel_jacobi_pcg.cu).
// Reuses build_poisson_csr from
// ../1_sparse_rewrite/kernel_sparse.cu verbatim for the matrix.
//
// One-time setup: factor A ~= L*U in place via cusparseXcsrilu02 (on a
// COPY of A's values -- the original dev_val_A stays untouched for the
// ongoing Ap matvec, since PCG needs both A and its factored M=LU every
// iteration), wrap L/U as cusparseSpMatDescr_t with fill-mode/diag-type
// attributes, run cusparseSpSV_analysis once. Per iteration: one
// cusparseSpMV for Ap, plus two cusparseSpSV_solve calls (forward on L,
// backward on U) to get z = M^-1 r = (LU)^-1 r.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "device_launch_parameters.h"

#ifndef FULLN
#define FULLN 66
#endif
#define TOL 1e-6
#define MAX_ITER 5000

#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s (at %s:%d)\n", cudaGetErrorString(cudaCheckReturn_e), __FILE__, __LINE__); \
      fflush(stderr); \
      exit(1); \
    } \
  } while(0)

#define cusparseCheckReturn(ret) \
  do { \
    cusparseStatus_t cusparseCheckReturn_e = (ret); \
    if (cusparseCheckReturn_e != CUSPARSE_STATUS_SUCCESS) { \
      fprintf(stderr, "cuSPARSE error: %s (at %s:%d)\n", cusparseGetErrorString(cusparseCheckReturn_e), __FILE__, __LINE__); \
      fflush(stderr); \
      exit(1); \
    } \
  } while(0)

int build_poisson_csr(int m, int** row_ptr_out, int** col_idx_out, float** val_out) {
    int n = m * m;
    int* row_ptr = (int*)malloc((n + 1) * sizeof(int));
    int* col_idx = (int*)malloc(5 * n * sizeof(int));
    float* val = (float*)malloc(5 * n * sizeof(float));

    int nnz = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        int row_idx = i / m;
        int col_in_row = i % m;

        if (row_idx > 0) { col_idx[nnz] = i - m; val[nnz] = -1.0f; nnz++; }
        if (col_in_row > 0) { col_idx[nnz] = i - 1; val[nnz] = -1.0f; nnz++; }
        col_idx[nnz] = i; val[nnz] = 4.0f; nnz++;
        if (col_in_row < m - 1) { col_idx[nnz] = i + 1; val[nnz] = -1.0f; nnz++; }
        if (row_idx < m - 1) { col_idx[nnz] = i + m; val[nnz] = -1.0f; nnz++; }

        row_ptr[i + 1] = nnz;
    }

    *row_ptr_out = row_ptr;
    *col_idx_out = col_idx;
    *val_out = val;
    return nnz;
}

int main() {
    int fulln = FULLN;
    int m = fulln - 2;
    int N = m * m;
    float xh, yh, h;
    h = 1.0f / (fulln - 1);

    int* row_ptr;
    int* col_idx;
    float* val;
    int nnz = build_poisson_csr(m, &row_ptr, &col_idx, &val);
    printf("N=%d, nnz=%d\n", N, nnz);

    float* W = (float*)malloc(fulln * fulln * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));
    float* x = (float*)calloc(N, sizeof(float));
    float* r = (float*)malloc(N * sizeof(float));
    float* z = (float*)malloc(N * sizeof(float));
    float* p = (float*)malloc(N * sizeof(float));
    float* Ap = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < fulln; i++) {
        xh = i * h;
        W[0 * fulln + i] = 1 + xh * xh;
        W[i * fulln + 0] = 1 + xh * xh;
        W[(fulln - 1) * fulln + i] = 1 + 1 + xh * xh;
        W[i * fulln + (fulln - 1)] = 1 + 1 + xh * xh;
    }
    for (int i = 0; i < m; i++) {
        xh = (i + 1) * h;
        for (int j = 0; j < m; j++) {
            yh = (j + 1) * h;
            b[i * m + j] = h * h * (-4);
            if (i == 0) { b[i * m + j] += W[0 * fulln + j + 1]; }
            if (i == m - 1) { b[i * m + j] += W[(i + 2) * fulln + j + 1]; }
            if (j == 0) { b[i * m + j] += W[(i + 1) * fulln]; }
            if (j == m - 1) { b[i * m + j] += W[(i + 1) * fulln + fulln - 1]; }
        }
    }
    printf("b[0]: %.3f\n", b[0]);

    // ---- device CSR for A (unchanged, used for Ap every iteration) ----
    int* dev_row_ptr; int* dev_col_idx; float* dev_val_A;
    cudaCheckReturn(cudaMalloc((void**)&dev_row_ptr, (N + 1) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void**)&dev_col_idx, nnz * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void**)&dev_val_A, nnz * sizeof(float)));
    cudaCheckReturn(cudaMemcpy(dev_row_ptr, row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_val_A, val, nnz * sizeof(float), cudaMemcpyHostToDevice));

    // ---- separate device copy of values, factored in place into L+U ----
    float* dev_val_ILU;
    cudaCheckReturn(cudaMalloc((void**)&dev_val_ILU, nnz * sizeof(float)));
    cudaCheckReturn(cudaMemcpy(dev_val_ILU, val, nnz * sizeof(float), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    cusparseCheckReturn(cusparseCreate(&handle));

    // ---- ILU(0) factorization (legacy csrilu02 API) ----
    cusparseMatDescr_t descrA;
    cusparseCheckReturn(cusparseCreateMatDescr(&descrA));
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    csrilu02Info_t infoILU;
    cusparseCheckReturn(cusparseCreateCsrilu02Info(&infoILU));

    int bufSizeILU = 0;
    cusparseCheckReturn(cusparseScsrilu02_bufferSize(
        handle, N, nnz, descrA, dev_val_ILU, dev_row_ptr, dev_col_idx, infoILU, &bufSizeILU));
    void* pBufferILU;
    cudaCheckReturn(cudaMalloc(&pBufferILU, bufSizeILU));

    cusparseCheckReturn(cusparseScsrilu02_analysis(
        handle, N, nnz, descrA, dev_val_ILU, dev_row_ptr, dev_col_idx, infoILU,
        CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBufferILU));
    int zeroPivot;
    cusparseStatus_t pivStatus = cusparseXcsrilu02_zeroPivot(handle, infoILU, &zeroPivot);
    if (pivStatus == CUSPARSE_STATUS_ZERO_PIVOT) {
        printf("ILU0 analysis: zero pivot at row %d\n", zeroPivot);
    }

    cusparseCheckReturn(cusparseScsrilu02(
        handle, N, nnz, descrA, dev_val_ILU, dev_row_ptr, dev_col_idx, infoILU,
        CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBufferILU));
    pivStatus = cusparseXcsrilu02_zeroPivot(handle, infoILU, &zeroPivot);
    if (pivStatus == CUSPARSE_STATUS_ZERO_PIVOT) {
        printf("ILU0 factorization: zero pivot at row %d\n", zeroPivot);
        return -1;
    }
    printf("ILU(0) factorization complete, no zero pivots.\n");

    // ---- wrap L (unit lower) and U (non-unit upper) sharing dev_val_ILU ----
    cusparseSpMatDescr_t matL, matU;
    cusparseCheckReturn(cusparseCreateCsr(
        &matL, N, N, nnz, dev_row_ptr, dev_col_idx, dev_val_ILU,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseFillMode_t fillLower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diagUnit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseCheckReturn(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE, &fillLower, sizeof(fillLower)));
    cusparseCheckReturn(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE, &diagUnit, sizeof(diagUnit)));

    cusparseCheckReturn(cusparseCreateCsr(
        &matU, N, N, nnz, dev_row_ptr, dev_col_idx, dev_val_ILU,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseFillMode_t fillUpper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diagNonUnit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseCheckReturn(cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_FILL_MODE, &fillUpper, sizeof(fillUpper)));
    cusparseCheckReturn(cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_DIAG_TYPE, &diagNonUnit, sizeof(diagNonUnit)));

    // ---- device vectors for the PCG loop ----
    float *dev_p, *dev_Ap, *dev_r, *dev_y, *dev_z;
    cudaCheckReturn(cudaMalloc(&dev_p, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc(&dev_Ap, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc(&dev_r, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc(&dev_y, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc(&dev_z, N * sizeof(float)));

    cusparseSpMatDescr_t matA;
    cusparseCheckReturn(cusparseCreateCsr(
        &matA, N, N, nnz, dev_row_ptr, dev_col_idx, dev_val_A,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnVecDescr_t vecP, vecAp, vecR, vecY, vecZ;
    cusparseCheckReturn(cusparseCreateDnVec(&vecP, N, dev_p, CUDA_R_32F));
    cusparseCheckReturn(cusparseCreateDnVec(&vecAp, N, dev_Ap, CUDA_R_32F));
    cusparseCheckReturn(cusparseCreateDnVec(&vecR, N, dev_r, CUDA_R_32F));
    cusparseCheckReturn(cusparseCreateDnVec(&vecY, N, dev_y, CUDA_R_32F));
    cusparseCheckReturn(cusparseCreateDnVec(&vecZ, N, dev_z, CUDA_R_32F));

    const float one = 1.0f, zero = 0.0f;
    size_t bufSizeMV = 0;
    cusparseCheckReturn(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecP, &zero, vecAp,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSizeMV));
    void* dBufferMV; cudaCheckReturn(cudaMalloc(&dBufferMV, bufSizeMV));

    // ---- SpSV setup: L*y = r, then U*z = y ----
    cusparseSpSVDescr_t spsvL, spsvU;
    cusparseCheckReturn(cusparseSpSV_createDescr(&spsvL));
    cusparseCheckReturn(cusparseSpSV_createDescr(&spsvU));

    size_t bufSizeL = 0, bufSizeU = 0;
    cusparseCheckReturn(cusparseSpSV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, vecR, vecY,
        CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvL, &bufSizeL));
    void* dBufferL; cudaCheckReturn(cudaMalloc(&dBufferL, bufSizeL));
    cusparseCheckReturn(cusparseSpSV_analysis(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, vecR, vecY,
        CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvL, dBufferL));

    cusparseCheckReturn(cusparseSpSV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matU, vecY, vecZ,
        CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvU, &bufSizeU));
    void* dBufferU; cudaCheckReturn(cudaMalloc(&dBufferU, bufSizeU));
    cusparseCheckReturn(cusparseSpSV_analysis(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matU, vecY, vecZ,
        CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvU, dBufferU));

    // ---- PCG initialization: r0 = b (x0=0), z0 = (LU)^-1 r0 ----
    for (int i = 0; i < N; i++) r[i] = b[i];
    cudaCheckReturn(cudaMemcpy(dev_r, r, N * sizeof(float), cudaMemcpyHostToDevice));
    cusparseCheckReturn(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, vecR, vecY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvL));
    cusparseCheckReturn(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matU, vecY, vecZ, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvU));
    cudaCheckReturn(cudaMemcpy(z, dev_z, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) p[i] = z[i];

    float rz_old = 0.0f;
    for (int i = 0; i < N; i++) rz_old += r[i] * z[i];

    clock_t start_time = clock();
    int iters = 0;

    for (int k = 0; k < MAX_ITER; k++) {
        iters = k + 1;
        cudaCheckReturn(cudaMemcpy(dev_p, p, N * sizeof(float), cudaMemcpyHostToDevice));

        cusparseCheckReturn(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecP, &zero, vecAp,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBufferMV));
        cudaCheckReturn(cudaMemcpy(Ap, dev_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));

        float pAp = 0.0f;
        for (int i = 0; i < N; i++) pAp += p[i] * Ap[i];
        float alpha = rz_old / pAp;

        for (int i = 0; i < N; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        float rnorm2 = 0.0f;
        for (int i = 0; i < N; i++) rnorm2 += r[i] * r[i];
        if (sqrt(rnorm2) < TOL) break;

        cudaCheckReturn(cudaMemcpy(dev_r, r, N * sizeof(float), cudaMemcpyHostToDevice));
        cusparseCheckReturn(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, vecR, vecY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvL));
        cusparseCheckReturn(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matU, vecY, vecZ, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvU));
        cudaCheckReturn(cudaMemcpy(z, dev_z, N * sizeof(float), cudaMemcpyDeviceToHost));

        float rz_new = 0.0f;
        for (int i = 0; i < N; i++) rz_new += r[i] * z[i];
        float beta = rz_new / rz_old;

        for (int i = 0; i < N; i++) p[i] = z[i] + beta * p[i];
        rz_old = rz_new;
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) * 1000.0 / CLOCKS_PER_SEC;

    float final_rnorm2 = 0.0f;
    for (int i = 0; i < N; i++) final_rnorm2 += r[i] * r[i];
    printf("Converged in %d iterations, final residual norm: %.6e\n", iters, sqrt((double)final_rnorm2));

    for (int i = 1; i < fulln - 1; i++)
        for (int j = 1; j < fulln - 1; j++)
            W[i * fulln + j] = x[j + (i - 1) * m - 1];
    double max_err = 0.0;
    for (int i = 1; i < fulln - 1; i++) {
        xh = i * h;
        for (int j = 1; j < fulln - 1; j++) {
            yh = j * h;
            double exact = 1 + xh * xh + yh * yh;
            double err = fabs(W[i * fulln + j] - exact);
            if (err > max_err) max_err = err;
        }
    }
    printf("Max abs error vs analytical solution: %.6e\n", max_err);
    printf("Time taken for main loop: %.3f ms\n", elapsed_time);

    cusparseSpSV_destroyDescr(spsvL);
    cusparseSpSV_destroyDescr(spsvU);
    cusparseDestroySpMat(matA); cusparseDestroySpMat(matL); cusparseDestroySpMat(matU);
    cusparseDestroyDnVec(vecP); cusparseDestroyDnVec(vecAp); cusparseDestroyDnVec(vecR);
    cusparseDestroyDnVec(vecY); cusparseDestroyDnVec(vecZ);
    cusparseDestroyCsrilu02Info(infoILU);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(handle);
    cudaFree(dBufferMV); cudaFree(dBufferL); cudaFree(dBufferU); cudaFree(pBufferILU);
    cudaFree(dev_row_ptr); cudaFree(dev_col_idx); cudaFree(dev_val_A); cudaFree(dev_val_ILU);
    cudaFree(dev_p); cudaFree(dev_Ap); cudaFree(dev_r); cudaFree(dev_y); cudaFree(dev_z);
    free(row_ptr); free(col_idx); free(val);
    free(W); free(b); free(x); free(r); free(z); free(p); free(Ap);

    return 0;
}
