#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cusparse.h>

// Generalized version of ../../4_cusparse/kernel_cusparse.cu for a
// ROWS x COLS interior grid.
#ifndef ROWS
#error "ROWS must be defined"
#endif
#ifndef COLS
#error "COLS must be defined"
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

#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
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

int build_poisson_csr(int rows, int cols, int** row_ptr_out, int** col_idx_out, float** val_out) {
    int n = rows * cols;
    int* row_ptr = (int*)malloc((n + 1) * sizeof(int));
    int* col_idx = (int*)malloc(5 * n * sizeof(int));
    float* val = (float*)malloc(5 * n * sizeof(float));
    int nnz = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        int row_idx = i / cols, col_in_row = i % cols;
        if (row_idx > 0) { col_idx[nnz] = i - cols; val[nnz] = -1.0f; nnz++; }
        if (col_in_row > 0) { col_idx[nnz] = i - 1; val[nnz] = -1.0f; nnz++; }
        col_idx[nnz] = i; val[nnz] = 4.0f; nnz++;
        if (col_in_row < cols - 1) { col_idx[nnz] = i + 1; val[nnz] = -1.0f; nnz++; }
        if (row_idx < rows - 1) { col_idx[nnz] = i + cols; val[nnz] = -1.0f; nnz++; }
        row_ptr[i + 1] = nnz;
    }
    *row_ptr_out = row_ptr; *col_idx_out = col_idx; *val_out = val;
    return nnz;
}

int main() {
    const int R = ROWS, C = COLS, N = R * C;
    const int fulln_r = R + 2, fulln_c = C + 2;
    float h = 1.0f / (fulln_c - 1);

    int* row_ptr; int* col_idx; float* val;
    int nnz = build_poisson_csr(R, C, &row_ptr, &col_idx, &val);
    printf("N=%d (%dx%d), nnz=%d\n", N, R, C, nnz);

    float* W = (float*)malloc(fulln_r * fulln_c * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));
    float* x = (float*)calloc(N, sizeof(float));
    float* r = (float*)malloc(N * sizeof(float));
    float* p = (float*)malloc(N * sizeof(float));
    float* Ap = (float*)malloc(N * sizeof(float));

    for (int col = 0; col < fulln_c; col++) {
        float x_ = col * h;
        W[0 * fulln_c + col] = 1 + x_ * x_;
        W[(fulln_r - 1) * fulln_c + col] = 1 + x_ * x_ + ((fulln_r - 1) * h) * ((fulln_r - 1) * h);
    }
    for (int row = 0; row < fulln_r; row++) {
        float y_ = row * h;
        W[row * fulln_c + 0] = 1 + y_ * y_;
        W[row * fulln_c + (fulln_c - 1)] = 1 + ((fulln_c - 1) * h) * ((fulln_c - 1) * h) + y_ * y_;
    }
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            b[i * C + j] = h * h * (-4);
            if (i == 0) b[i * C + j] += W[0 * fulln_c + (j + 1)];
            if (i == R - 1) b[i * C + j] += W[(fulln_r - 1) * fulln_c + (j + 1)];
            if (j == 0) b[i * C + j] += W[(i + 1) * fulln_c + 0];
            if (j == C - 1) b[i * C + j] += W[(i + 1) * fulln_c + (fulln_c - 1)];
        }
    }

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
    cusparseCheckReturn(cusparseCreateCsr(&matA, N, N, nnz, dev_row_ptr, dev_col_idx, dev_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnVecDescr_t vecP, vecAp;
    cusparseCheckReturn(cusparseCreateDnVec(&vecP, N, dev_p, CUDA_R_32F));
    cusparseCheckReturn(cusparseCreateDnVec(&vecAp, N, dev_Ap, CUDA_R_32F));
    const float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    cusparseCheckReturn(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecP, &beta, vecAp, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    void* dBuffer = NULL;
    cudaCheckReturn(cudaMalloc(&dBuffer, bufferSize));

    for (int i = 0; i < N; i++) { r[i] = b[i]; p[i] = r[i]; }
    float rs_old = 0.0f;
    for (int i = 0; i < N; i++) rs_old += r[i] * r[i];

    clock_t start_time = clock();
    int iters = 0;
    for (int k = 0; k < MAX_ITER; k++) {
        iters = k + 1;
        cudaCheckReturn(cudaMemcpy(dev_p, p, N * sizeof(float), cudaMemcpyHostToDevice));
        cusparseCheckReturn(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecP, &beta, vecAp, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        cudaCheckReturn(cudaMemcpy(Ap, dev_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));

        float pAp = 0.0f;
        for (int i = 0; i < N; i++) pAp += p[i] * Ap[i];
        float alpha_cg = rs_old / pAp;
        for (int i = 0; i < N; i++) { x[i] += alpha_cg * p[i]; r[i] -= alpha_cg * Ap[i]; }
        float rs_new = 0.0f;
        for (int i = 0; i < N; i++) rs_new += r[i] * r[i];
        if (sqrt(rs_new) < TOL) { rs_old = rs_new; break; }
        float beta_cg = rs_new / rs_old;
        for (int i = 0; i < N; i++) p[i] = r[i] + beta_cg * p[i];
        rs_old = rs_new;
    }
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) * 1000.0 / CLOCKS_PER_SEC;

    printf("Converged in %d iterations, final residual norm: %.6e\n", iters, sqrt((double)rs_old));

    double max_err = 0.0;
    for (int i = 0; i < R; i++) {
        float y_ = (i + 1) * h;
        for (int j = 0; j < C; j++) {
            float x_ = (j + 1) * h;
            double exact = 1 + (double)x_ * x_ + (double)y_ * y_;
            double err = fabs((double)x[i * C + j] - exact);
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
    free(W); free(b); free(x); free(r); free(p); free(Ap);
    return 0;
}
