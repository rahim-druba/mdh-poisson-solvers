#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Same test problem as kernel2 (2).cu: fulln=66 -> interior grid is 64x64 -> N=4096.
// Change FULLN to scale the problem for later size-sweep comparisons.
#define FULLN 66
#define TOL 1e-6
#define MAX_ITER 5000

// ==== CUDA error-check macros (same as kernel2 (2).cu) ====
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
// ============================================================

// Naive CSR-scalar SpMV: one thread per row. This is the correctness
// baseline -- not the MDH-tuned kernel. That comes in step 2.
__global__ void spmv_csr_kernel(int n, const int* __restrict__ row_ptr,
                                 const int* __restrict__ col_idx,
                                 const float* __restrict__ val,
                                 const float* __restrict__ p,
                                 float* __restrict__ Ap) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        for (int k = start; k < end; ++k) {
            sum += val[k] * p[col_idx[k]];
        }
        Ap[row] = sum;
    }
}

// Builds the CSR representation of the 2D 5-point Poisson stencil directly,
// without ever forming the dense N x N matrix. Reproduces exactly the
// sparsity pattern that kernel2 (2).cu built densely (diag=4, neighbors=-1,
// with the same row-boundary correction to avoid spurious wrap-around
// connections between grid rows).
int build_poisson_csr(int m /* = FULLN-2, grid side length */,
                       int** row_ptr_out, int** col_idx_out, float** val_out) {
    int n = m * m;
    int* row_ptr = (int*)malloc((n + 1) * sizeof(int));
    // upper bound: 5 nonzeros per row
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
    int m = fulln - 2;          // interior grid side length
    int N = m * m;               // linear system size
    float xh, yh, h;
    h = 1.0f / (fulln - 1);

    // ---- Build CSR matrix (host) ----
    int* row_ptr;
    int* col_idx;
    float* val;
    int nnz = build_poisson_csr(m, &row_ptr, &col_idx, &val);

    double dense_bytes = (double)N * (double)N * sizeof(float);
    double sparse_bytes = (double)nnz * (sizeof(float) + sizeof(int)) + (double)(N + 1) * sizeof(int);
    printf("N=%d, nnz=%d (%.4f%% dense)\n", N, nnz, 100.0 * nnz / ((double)N * N));
    printf("Dense storage would need %.2f MB, CSR needs %.2f MB\n",
           dense_bytes / (1024.0 * 1024.0), sparse_bytes / (1024.0 * 1024.0));

    // ---- Boundary values W and right-hand side b (same construction as kernel2 (2).cu) ----
    float* W = (float*)malloc(fulln * fulln * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));
    float* x = (float*)calloc(N, sizeof(float));
    float* r = (float*)malloc(N * sizeof(float));
    float* p = (float*)malloc(N * sizeof(float));
    float* Ap = (float*)malloc(N * sizeof(float));

    if (!W || !b || !x || !r || !p || !Ap) {
        printf("Allocation failure\n");
        return -1;
    }

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

    // ---- GPU CSR buffers ----
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

    // ---- CG initialization: r = b - A*x, x = 0 so r = b ----
    for (int i = 0; i < N; i++) {
        r[i] = b[i];
        p[i] = r[i];
    }

    float rs_old = 0.0f;
    for (int i = 0; i < N; i++) rs_old += r[i] * r[i];
    printf("rs_old: %.3f\n", rs_old);

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

        for (int i = 0; i < N; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

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

    // ---- Verify against analytical solution u(x,y) = 1 + x^2 + y^2 ----
    for (int i = 1; i < fulln - 1; i++) {
        for (int j = 1; j < fulln - 1; j++) {
            W[i * fulln + j] = x[j + (i - 1) * m - 1];
        }
    }
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

    cudaFree(dev_row_ptr); cudaFree(dev_col_idx); cudaFree(dev_val);
    cudaFree(dev_p); cudaFree(dev_Ap);
    free(row_ptr); free(col_idx); free(val);
    free(W); free(b); free(x); free(r); free(p); free(Ap);

    return 0;
}
