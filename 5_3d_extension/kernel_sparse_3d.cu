// Build (M = interior grid side length, e.g. 8/16/24/32):
//   nvcc -O3 kernel_sparse_3d.cu -o kernel_sparse_3d_M16 -DM=16
// Run:
//   ./kernel_sparse_3d_M16
//
// Hand-written CSR 3D solver -- direct 3D extension of
// ../1_sparse_rewrite/kernel_sparse.cu. 3D Poisson equation on the unit
// cube, Dirichlet BC, analytical solution u(x,y,z) = 1 + x^2 + y^2 + z^2
// (direct extension of the 2D u = 1 + x^2 + y^2 used everywhere else in
// this project), so -Laplacian(u) = -6 (constant), 7-point stencil
// (diag=6, six face-neighbors -1).

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#ifndef M
#error "M must be defined (-DM=.. interior grid side length)"
#endif
#define FULLN (M + 2)
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

static inline float u_exact(float x, float y, float z) {
    return 1.0f + x * x + y * y + z * z;
}

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

// Builds the CSR representation of the 3D 7-point Poisson stencil directly,
// without ever forming the dense N x N matrix (diag=6, six face-neighbors
// -1, boundary rows omit the missing neighbor entries -- equivalent
// to Dirichlet oob=0). Direct 3D extension of kernel_sparse.cu's
// build_poisson_csr.
int build_poisson_csr_3d(int m, int** row_ptr_out, int** col_idx_out, float** val_out) {
    int n = m * m * m;
    int* row_ptr = (int*)malloc((n + 1) * sizeof(int));
    int* col_idx = (int*)malloc(7 * n * sizeof(int));
    float* val = (float*)malloc(7 * n * sizeof(float));

    int nnz = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < m; ++k) {
                int idx = (i * m + j) * m + k;

                if (i > 0)     { col_idx[nnz] = ((i - 1) * m + j) * m + k; val[nnz] = -1.0f; nnz++; }
                if (j > 0)     { col_idx[nnz] = (i * m + (j - 1)) * m + k; val[nnz] = -1.0f; nnz++; }
                if (k > 0)     { col_idx[nnz] = (i * m + j) * m + (k - 1); val[nnz] = -1.0f; nnz++; }
                col_idx[nnz] = idx; val[nnz] = 6.0f; nnz++;
                if (k < m - 1) { col_idx[nnz] = (i * m + j) * m + (k + 1); val[nnz] = -1.0f; nnz++; }
                if (j < m - 1) { col_idx[nnz] = (i * m + (j + 1)) * m + k; val[nnz] = -1.0f; nnz++; }
                if (i < m - 1) { col_idx[nnz] = ((i + 1) * m + j) * m + k; val[nnz] = -1.0f; nnz++; }

                row_ptr[idx + 1] = nnz;
            }
        }
    }

    *row_ptr_out = row_ptr;
    *col_idx_out = col_idx;
    *val_out = val;
    return nnz;
}

int main() {
    int fulln = FULLN;
    int m = fulln - 2;          // interior grid side length
    int N = m * m * m;
    float h = 1.0f / (fulln - 1);

    int* row_ptr;
    int* col_idx;
    float* val;
    int nnz = build_poisson_csr_3d(m, &row_ptr, &col_idx, &val);

    double dense_bytes = (double)N * (double)N * sizeof(float);
    double sparse_bytes = (double)nnz * (sizeof(float) + sizeof(int)) + (double)(N + 1) * sizeof(int);
    printf("N=%d, nnz=%d (%.4f%% dense)\n", N, nnz, 100.0 * nnz / ((double)N * N));
    printf("Dense storage would need %.2f MB, CSR needs %.2f MB\n",
           dense_bytes / (1024.0 * 1024.0), sparse_bytes / (1024.0 * 1024.0));

    float* b = (float*)malloc(N * sizeof(float));
    float* x = (float*)calloc(N, sizeof(float));
    float* r = (float*)malloc(N * sizeof(float));
    float* p = (float*)malloc(N * sizeof(float));
    float* Ap = (float*)malloc(N * sizeof(float));

    if (!b || !x || !r || !p || !Ap) {
        printf("Allocation failure\n");
        return -1;
    }

    // ---- Right-hand side b: -Laplacian(u) = -6, plus boundary contributions
    // from any of the 6 cube faces the interior node touches ----
    for (int i = 0; i < m; i++) {
        float xh = (i + 1) * h;
        for (int j = 0; j < m; j++) {
            float yh = (j + 1) * h;
            for (int k = 0; k < m; k++) {
                float zh = (k + 1) * h;
                int idx = (i * m + j) * m + k;
                b[idx] = h * h * (-6.0f);
                if (i == 0)     b[idx] += u_exact(0.0f, yh, zh);
                if (i == m - 1) b[idx] += u_exact(1.0f, yh, zh);
                if (j == 0)     b[idx] += u_exact(xh, 0.0f, zh);
                if (j == m - 1) b[idx] += u_exact(xh, 1.0f, zh);
                if (k == 0)     b[idx] += u_exact(xh, yh, 0.0f);
                if (k == m - 1) b[idx] += u_exact(xh, yh, 1.0f);
            }
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

    for (int i = 0; i < N; i++) { r[i] = b[i]; p[i] = r[i]; }

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

    // ---- Verify against analytical solution u(x,y,z) = 1 + x^2 + y^2 + z^2 ----
    double max_err = 0.0;
    for (int i = 0; i < m; i++) {
        float xh = (i + 1) * h;
        for (int j = 0; j < m; j++) {
            float yh = (j + 1) * h;
            for (int k = 0; k < m; k++) {
                float zh = (k + 1) * h;
                int idx = (i * m + j) * m + k;
                double exact = u_exact(xh, yh, zh);
                double err = fabs((double)x[idx] - exact);
                if (err > max_err) max_err = err;
            }
        }
    }
    printf("Max abs error vs analytical solution: %.6e\n", max_err);
    printf("Time taken for main loop: %.3f ms\n", elapsed_time);

    cudaFree(dev_row_ptr); cudaFree(dev_col_idx); cudaFree(dev_val);
    cudaFree(dev_p); cudaFree(dev_Ap);
    free(row_ptr); free(col_idx); free(val);
    free(b); free(x); free(r); free(p); free(Ap);

    return 0;
}
