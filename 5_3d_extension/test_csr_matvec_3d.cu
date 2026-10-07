// Hand-written CSR 3D baseline: builds the 3D 7-point Poisson stencil as a
// real CSR matrix (up to 7 nonzeros/row instead of the 2D case's 5), then
// runs the same one-thread-per-row SpMV kernel as
// ../1_sparse_rewrite/kernel_sparse.cu, verified against the CPU reference.
//
// Build:
//   nvcc -O3 test_csr_matvec_3d.cu -o test_csr_matvec_3d
// Run:
//   ./test_csr_matvec_3d

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cpu_reference_matvec_3d.h"

#define M 16

#define cudaCheckReturn(ret) \
  do { \
    cudaError_t e = (ret); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s (at %s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

// Same shape as 1_sparse_rewrite/kernel_sparse.cu's spmv_csr_kernel --
// one thread per row, agnostic to dimensionality.
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
// without ever forming the dense N x N matrix. Direct 3D extension of
// 1_sparse_rewrite/kernel_sparse.cu's build_poisson_csr (diag=6, six
// face-neighbors -1, with boundary rows omitting the missing
// neighbor entries -- equivalent to Dirichlet oob=0).
int build_poisson_csr_3d(int m, int** row_ptr_out, int** col_idx_out, float** val_out) {
    int n = m * m * m;
    int* row_ptr = (int*)malloc((n + 1) * sizeof(int));
    int* col_idx = (int*)malloc(7 * n * sizeof(int)); // upper bound: 7 nonzeros/row
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
    const int m = M;
    const int N = m * m * m;

    int* row_ptr;
    int* col_idx;
    float* val;
    int nnz = build_poisson_csr_3d(m, &row_ptr, &col_idx, &val);
    printf("N=%d, nnz=%d (%.4f%% dense)\n", N, nnz, 100.0 * nnz / ((double)N * N));

    float* p = (float*)malloc(N * sizeof(float));
    float* Ap_cpu = (float*)malloc(N * sizeof(float));
    float* Ap_gpu = (float*)malloc(N * sizeof(float));

    srand(0);
    for (int i = 0; i < N; ++i) p[i] = (float)(rand() % 100) / 10.0f - 5.0f;

    cpu_matvec_3d(m, p, Ap_cpu);

    int *dev_row_ptr, *dev_col_idx;
    float *dev_val, *dev_p, *dev_Ap;
    cudaCheckReturn(cudaMalloc((void**)&dev_row_ptr, (N + 1) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void**)&dev_col_idx, nnz * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void**)&dev_val, nnz * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_p, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_Ap, N * sizeof(float)));

    cudaCheckReturn(cudaMemcpy(dev_row_ptr, row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_val, val, nnz * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_p, p, N * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (N + block - 1) / block;
    spmv_csr_kernel<<<grid, block>>>(N, dev_row_ptr, dev_col_idx, dev_val, dev_p, dev_Ap);
    cudaCheckReturn(cudaGetLastError());
    cudaCheckReturn(cudaDeviceSynchronize());

    cudaCheckReturn(cudaMemcpy(Ap_gpu, dev_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = fabsf(Ap_gpu[i] - Ap_cpu[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-4f) {
            if (mismatches < 10) printf("MISMATCH idx=%d gpu=%.6f cpu=%.6f diff=%.3e\n", i, Ap_gpu[i], Ap_cpu[i], err);
            mismatches++;
        }
    }
    printf("mismatches=%d, max_err=%.3e\n", mismatches, max_err);
    printf(mismatches == 0 ? "PASS\n" : "FAIL\n");

    cudaFree(dev_row_ptr); cudaFree(dev_col_idx); cudaFree(dev_val); cudaFree(dev_p); cudaFree(dev_Ap);
    free(row_ptr); free(col_idx); free(val); free(p); free(Ap_cpu); free(Ap_gpu);
    return mismatches == 0 ? 0 : 1;
}
