// Build:
//   nvcc -O3 kernel_jacobi_pcg.cu -o kernel_jacobi_pcg -lcusparse
// Run:
//   ./kernel_jacobi_pcg
//
// Jacobi-preconditioned CG (PCG). Direct extension of
// ../4_cusparse/kernel_cusparse.cu's plain CG loop with one added step:
// z_k = M^-1 * r_k, M = diag(A).
//
// Expected result, not assumed: the discrete 5-point Poisson operator has
// a CONSTANT diagonal (every interior row has diag=4), so M = 4I exactly
// -- a uniform scalar. Preconditioned CG with M = cI for any constant c is
// algebraically identical to unpreconditioned CG (the scaling cancels in
// the alpha/beta ratios): z_k = r_k/4 throughout, so p_k is just a
// constant multiple of what plain CG's search direction would be at every
// step, and the iteration count should come out EXACTLY equal to the
// existing plain-CG baseline (kernel_cusparse.cu), not fewer. This file
// exists to confirm that with real numbers, not to assume it.

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

// Identical to 1_sparse_rewrite/kernel_sparse.cu's build_poisson_csr.
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

    int* dev_row_ptr; int* dev_col_idx; float* dev_val;
    float* dev_p; float* dev_Ap;
    cudaCheckReturn(cudaMalloc((void**)&dev_row_ptr, (N + 1) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void**)&dev_col_idx, nnz * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void**)&dev_val, nnz * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_p, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_Ap, N * sizeof(float)));

    cudaCheckReturn(cudaMemcpy(dev_row_ptr, row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_val, val, nnz * sizeof(float), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    cusparseCheckReturn(cusparseCreate(&handle));
    cusparseSpMatDescr_t matA;
    cusparseCheckReturn(cusparseCreateCsr(
        &matA, N, N, nnz,
        dev_row_ptr, dev_col_idx, dev_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnVecDescr_t vecP, vecAp;
    cusparseCheckReturn(cusparseCreateDnVec(&vecP, N, dev_p, CUDA_R_32F));
    cusparseCheckReturn(cusparseCreateDnVec(&vecAp, N, dev_Ap, CUDA_R_32F));

    const float alpha_one = 1.0f, beta_zero = 0.0f;
    size_t bufferSize = 0;
    cusparseCheckReturn(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_one, matA, vecP, &beta_zero, vecAp,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    void* dBuffer = NULL;
    cudaCheckReturn(cudaMalloc(&dBuffer, bufferSize));

    // ---- PCG initialization: r0 = b - A*x0 = b (x0=0), z0 = M^-1 r0 = r0/4 ----
    for (int i = 0; i < N; i++) {
        r[i] = b[i];
        z[i] = r[i] / 4.0f;
        p[i] = z[i];
    }

    float rz_old = 0.0f;
    for (int i = 0; i < N; i++) rz_old += r[i] * z[i];

    clock_t start_time = clock();
    int iters = 0;

    for (int k = 0; k < MAX_ITER; k++) {
        iters = k + 1;
        cudaCheckReturn(cudaMemcpy(dev_p, p, N * sizeof(float), cudaMemcpyHostToDevice));

        cusparseCheckReturn(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_one, matA, vecP, &beta_zero, vecAp,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
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

        for (int i = 0; i < N; i++) z[i] = r[i] / 4.0f;

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

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecP);
    cusparseDestroyDnVec(vecAp);
    cusparseDestroy(handle);
    cudaFree(dBuffer);
    cudaFree(dev_row_ptr); cudaFree(dev_col_idx); cudaFree(dev_val);
    cudaFree(dev_p); cudaFree(dev_Ap);
    free(row_ptr); free(col_idx); free(val);
    free(W); free(b); free(x); free(r); free(z); free(p); free(Ap);

    return 0;
}
