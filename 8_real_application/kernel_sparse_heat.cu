// Build:
//   nvcc -O3 kernel_sparse_heat.cu -o kernel_sparse_heat
// Run:
//   ./kernel_sparse_heat
//
// Real-world application demo: steady-state heat conduction on a plate
// with a localized heat source. Same PDE (-Laplacian(u) = f) and CSR
// solver structure as
// ../1_sparse_rewrite/kernel_sparse.cu, but with a physically-motivated
// source and boundary instead of the synthetic verification problem:
//   f(x,y) = A * exp(-((x-0.5)^2+(y-0.5)^2)/(2*sigma^2))  -- a heating
//            element at the plate's center (A=100, sigma=0.1)
//   g(x,y) = 0 on all four edges -- frame held at reference temperature
//
// No closed-form solution exists for a Gaussian source on a bounded
// square (unlike the synthetic verification problem), so this writes the
// converged temperature field to a text file for cross-verification
// against ../8_real_application/kernel_mdh_heat.cu's independently
// computed solution (see verify_and_plot.py) instead of comparing to an
// analytical answer.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define FULLN 66
#define TOL 1e-6
#define MAX_ITER 5000
#define SOURCE_AMPLITUDE 100.0f
#define SOURCE_SIGMA 0.1f

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

static inline float heat_source(float x, float y) {
    float dx = x - 0.5f, dy = y - 0.5f;
    return SOURCE_AMPLITUDE * expf(-(dx * dx + dy * dy) / (2.0f * SOURCE_SIGMA * SOURCE_SIGMA));
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
        for (int k = start; k < end; ++k) sum += val[k] * p[col_idx[k]];
        Ap[row] = sum;
    }
}

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
    float h = 1.0f / (fulln - 1);

    int* row_ptr;
    int* col_idx;
    float* val;
    int nnz = build_poisson_csr(m, &row_ptr, &col_idx, &val);
    printf("N=%d, nnz=%d -- heat conduction, Gaussian source at plate center\n", N, nnz);

    float* b = (float*)malloc(N * sizeof(float));
    float* x = (float*)calloc(N, sizeof(float));
    float* r = (float*)malloc(N * sizeof(float));
    float* p = (float*)malloc(N * sizeof(float));
    float* Ap = (float*)malloc(N * sizeof(float));

    // ---- RHS: b = h^2 * f(x,y). Boundary g=0 everywhere, so no boundary
    // correction term is needed -- a missing neighbor at the edge already
    // implicitly contributes 0, exactly matching g=0. ----
    for (int i = 0; i < m; i++) {
        float xh = (i + 1) * h;
        for (int j = 0; j < m; j++) {
            float yh = (j + 1) * h;
            b[i * m + j] = h * h * heat_source(xh, yh);
        }
    }
    printf("b[center]: %.4f (h^2 * source), max source value: %.2f\n",
           b[(m / 2) * m + m / 2], SOURCE_AMPLITUDE);

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
    printf("Time taken for main loop: %.3f ms\n", elapsed_time);

    // ---- physical sanity checks ----
    float max_temp = -1e30f; int max_i = -1, max_j = -1;
    float min_temp = 1e30f;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            float v = x[i * m + j];
            if (v > max_temp) { max_temp = v; max_i = i; max_j = j; }
            if (v < min_temp) min_temp = v;
        }
    printf("Peak temperature %.4f at grid (%d,%d) -> physical (%.3f,%.3f) [expect near (0.5,0.5)]\n",
           max_temp, max_i, max_j, (max_i + 1) * h, (max_j + 1) * h);
    printf("Min temperature in interior: %.4f\n", min_temp);

    // ---- write solution grid for cross-verification / plotting ----
    FILE* f = fopen("solution_sparse.txt", "w");
    fprintf(f, "%d %d\n", m, m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) fprintf(f, "%.6f ", x[i * m + j]);
        fprintf(f, "\n");
    }
    fclose(f);
    printf("Wrote solution_sparse.txt (%dx%d grid)\n", m, m);

    cudaFree(dev_row_ptr); cudaFree(dev_col_idx); cudaFree(dev_val);
    cudaFree(dev_p); cudaFree(dev_Ap);
    free(row_ptr); free(col_idx); free(val);
    free(b); free(x); free(r); free(p); free(Ap);

    return 0;
}
