#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Generalized version of ../../1_sparse_rewrite/kernel_sparse.cu for a
// ROWS x COLS interior grid (non-square allowed). u_exact(x,y) = 1+x^2+y^2
// is used for validation on any rectangle with uniform spacing h in both
// directions (it satisfies -Laplacian(u) = -4 everywhere regardless of
// domain shape, so a rectangular, non-square grid is still exactly solvable
// and checkable).
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

__global__ void spmv_csr_kernel(int n, const int* __restrict__ row_ptr,
                                 const int* __restrict__ col_idx,
                                 const float* __restrict__ val,
                                 const float* __restrict__ p,
                                 float* __restrict__ Ap) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        int start = row_ptr[row], end = row_ptr[row + 1];
        for (int k = start; k < end; ++k) sum += val[k] * p[col_idx[k]];
        Ap[row] = sum;
    }
}

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

    // boundary values: u_exact(x=col*h, y=row*h) = 1 + x^2 + y^2
    for (int col = 0; col < fulln_c; col++) {
        float x_ = col * h;
        W[0 * fulln_c + col] = 1 + x_ * x_;                                  // top (row 0)
        W[(fulln_r - 1) * fulln_c + col] = 1 + x_ * x_ + ((fulln_r - 1) * h) * ((fulln_r - 1) * h); // bottom
    }
    for (int row = 0; row < fulln_r; row++) {
        float y_ = row * h;
        W[row * fulln_c + 0] = 1 + y_ * y_;                                  // left
        W[row * fulln_c + (fulln_c - 1)] = 1 + ((fulln_c - 1) * h) * ((fulln_c - 1) * h) + y_ * y_; // right
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

    for (int i = 0; i < N; i++) { r[i] = b[i]; p[i] = r[i]; }
    float rs_old = 0.0f;
    for (int i = 0; i < N; i++) rs_old += r[i] * r[i];

    int block = 256;
    int grid = (N + block - 1) / block;

    clock_t start_time = clock();
    int iters = 0;
    for (int k = 0; k < MAX_ITER; k++) {
        iters = k + 1;
        cudaCheckReturn(cudaMemcpy(dev_p, p, N * sizeof(float), cudaMemcpyHostToDevice));
        spmv_csr_kernel<<<grid, block>>>(N, dev_row_ptr, dev_col_idx, dev_val, dev_p, dev_Ap);
        cudaCheckKernel();
        cudaCheckReturn(cudaMemcpy(Ap, dev_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));

        float pAp = 0.0f;
        for (int i = 0; i < N; i++) pAp += p[i] * Ap[i];
        float alpha = rs_old / pAp;
        for (int i = 0; i < N; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
        float rs_new = 0.0f;
        for (int i = 0; i < N; i++) rs_new += r[i] * r[i];
        if (sqrt(rs_new) < TOL) { rs_old = rs_new; break; }
        float beta = rs_new / rs_old;
        for (int i = 0; i < N; i++) p[i] = r[i] + beta * p[i];
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

    cudaFree(dev_row_ptr); cudaFree(dev_col_idx); cudaFree(dev_val);
    cudaFree(dev_p); cudaFree(dev_Ap);
    free(row_ptr); free(col_idx); free(val);
    free(W); free(b); free(x); free(r); free(p); free(Ap);
    return 0;
}
