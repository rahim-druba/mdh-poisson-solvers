// cuSPARSE 3D baseline: true CSR storage (not matrix-free), reusing the
// same build_poisson_csr_3d builder as test_csr_matvec_3d.cu (duplicated
// here, same convention as ../4_cusparse/kernel_cusparse.cu duplicating
// kernel_sparse.cu's builder) and the generic cusparseSpMV API, verified
// against the CPU reference.
//
// Build:
//   nvcc -O3 test_cusparse_matvec_3d.cu -o test_cusparse_matvec_3d -lcusparse
// Run:
//   ./test_cusparse_matvec_3d

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cusparse.h>
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

#define cusparseCheckReturn(ret) \
  do { \
    cusparseStatus_t e = (ret); \
    if (e != CUSPARSE_STATUS_SUCCESS) { \
      fprintf(stderr, "cuSPARSE error: %s (at %s:%d)\n", cusparseGetErrorString(e), __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

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
    cudaCheckReturn(cudaMemcpy(dev_p, p, N * sizeof(float), cudaMemcpyHostToDevice));

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

    const float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    cusparseCheckReturn(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecP, &beta, vecAp,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    void* dBuffer = NULL;
    cudaCheckReturn(cudaMalloc(&dBuffer, bufferSize));

    cusparseCheckReturn(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecP, &beta, vecAp,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
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

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecP);
    cusparseDestroyDnVec(vecAp);
    cusparseDestroy(handle);
    cudaFree(dBuffer);
    cudaFree(dev_row_ptr); cudaFree(dev_col_idx); cudaFree(dev_val);
    cudaFree(dev_p); cudaFree(dev_Ap);
    free(row_ptr); free(col_idx); free(val); free(p); free(Ap_cpu); free(Ap_gpu);
    return mismatches == 0 ? 0 : 1;
}
