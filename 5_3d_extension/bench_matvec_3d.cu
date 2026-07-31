// 4-way isolated matvec comparison: Ap = A*p for the 3D 7-point Poisson
// stencil (diag=6, six face-neighbors -1), on a 16x16x16 interior grid
// (N=4096) -- the 3D counterpart of ../tables/matvec/bench_matvec.cu, same
// methodology (cudaEvent, time-based warmup, 200 timed launches).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "cg_matvec_3d_ppcg_src_kernel.hu"

#define CUDA_CHECK(e) do { \
    cudaError_t _err = (e); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUSPARSE_CHECK(e) do { \
    cusparseStatus_t _st = (e); \
    if (_st != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error '%s' at %s:%d\n", cusparseGetErrorString(_st), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#ifndef M
#error "M must be defined (-DM=.. interior grid side length)"
#endif
static const int N = M * M * M;
static const int PAD = M + 2;
static const int NP = PAD * PAD * PAD;

static const int TIMED = 200;
static const float MIN_WARMUP_MS = 300.0f;

extern __global__ void cg_matvec_3d_1(
    float const * const __restrict__ P,
    float       * const __restrict__ res_g,
    float       * const __restrict__ int_res
);

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

int build_poisson_csr_3d(int m, int** row_ptr_out, int** col_idx_out, float** val_out) {
    int n = m * m * m;
    int* row_ptr = (int*)malloc((n + 1) * sizeof(int));
    int* col_idx = (int*)malloc(7 * n * sizeof(int));
    float* val = (float*)malloc(7 * n * sizeof(float));
    int nnz = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
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
    *row_ptr_out = row_ptr; *col_idx_out = col_idx; *val_out = val;
    return nnz;
}

int main() {
    printf("=== 4-way 3D matvec comparison: Ap = A*p, N=%d (%dx%dx%d grid) ===\n", N, M, M, M);
    printf("Methodology: >=%.0fms time-based warmup + %d timed launches, cudaEvent, averaged.\n\n",
           MIN_WARMUP_MS, TIMED);

    float* h_p = (float*)malloc(N * sizeof(float));
    float* h_ref = (float*)malloc(N * sizeof(float));
    float* h_result = (float*)malloc(N * sizeof(float));
    float* h_p_padded = (float*)calloc(NP, sizeof(float));

    srand(0);
    for (int i = 0; i < N; i++) h_p[i] = (float)(rand() % 100) / 10.0f - 5.0f;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++)
            for (int k = 0; k < M; k++)
                h_p_padded[((i + 1) * PAD + (j + 1)) * PAD + (k + 1)] = h_p[(i * M + j) * M + k];

    // CPU reference (7-point stencil)
    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++)
            for (int k = 0; k < M; k++) {
                int idx = (i * M + j) * M + k;
                float sum = 6.0f * h_p[idx];
                if (i > 0)     sum -= h_p[((i - 1) * M + j) * M + k];
                if (i < M - 1) sum -= h_p[((i + 1) * M + j) * M + k];
                if (j > 0)     sum -= h_p[(i * M + (j - 1)) * M + k];
                if (j < M - 1) sum -= h_p[(i * M + (j + 1)) * M + k];
                if (k > 0)     sum -= h_p[(i * M + j) * M + (k - 1)];
                if (k < M - 1) sum -= h_p[(i * M + j) * M + (k + 1)];
                h_ref[idx] = sum;
            }

    int* row_ptr; int* col_idx; float* val;
    int nnz = build_poisson_csr_3d(M, &row_ptr, &col_idx, &val);

    float *d_p, *d_p_padded, *d_Ap, *d_res_g;
    int *d_row_ptr, *d_col_idx; float *d_val;
    CUDA_CHECK(cudaMalloc(&d_p, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_padded, NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ap, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_g, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, nnz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_p, h_p, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_padded, h_p_padded, NP * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, val, nnz * sizeof(float), cudaMemcpyHostToDevice));

    printf("%-10s %10s %10s %8s\n", "method", "avg ms", "max error", "correct");
    printf("--------------------------------------------------\n");

    // sparse (CSR)
    {
        dim3 grid((N + 255) / 256), block(256);
        cudaEvent_t ws, we; CUDA_CHECK(cudaEventCreate(&ws)); CUDA_CHECK(cudaEventCreate(&we));
        CUDA_CHECK(cudaEventRecord(ws));
        float warm_ms;
        do {
            spmv_csr_kernel<<<grid, block>>>(N, d_row_ptr, d_col_idx, d_val, d_p, d_Ap);
            CUDA_CHECK(cudaEventRecord(we)); CUDA_CHECK(cudaEventSynchronize(we));
            CUDA_CHECK(cudaEventElapsedTime(&warm_ms, ws, we));
        } while (warm_ms < MIN_WARMUP_MS);
        CUDA_CHECK(cudaEventDestroy(ws)); CUDA_CHECK(cudaEventDestroy(we));
        cudaEvent_t s, e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        for (int i = 0; i < TIMED; i++) spmv_csr_kernel<<<grid, block>>>(N, d_row_ptr, d_col_idx, d_val, d_p, d_Ap);
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); ms /= TIMED;
        CUDA_CHECK(cudaMemcpy(h_result, d_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int i = 0; i < N; i++) max_err = fmax(max_err, fabs((double)h_result[i] - h_ref[i]));
        printf("%-10s %10.5f %10.3e %8s\n", "sparse", ms, max_err, max_err < 1e-3 ? "yes" : "no");
    }

    // mdh (matrix-free)
    {
        dim3 block(NUM_WI_L_3, NUM_WI_L_2, NUM_WI_L_1);
        dim3 grid(NUM_WG_L_3, NUM_WG_L_2, NUM_WG_L_1);
        cudaEvent_t ws, we; CUDA_CHECK(cudaEventCreate(&ws)); CUDA_CHECK(cudaEventCreate(&we));
        CUDA_CHECK(cudaEventRecord(ws));
        float warm_ms;
        do {
            cg_matvec_3d_1<<<grid, block>>>(d_p, d_res_g, d_Ap);
            CUDA_CHECK(cudaEventRecord(we)); CUDA_CHECK(cudaEventSynchronize(we));
            CUDA_CHECK(cudaEventElapsedTime(&warm_ms, ws, we));
        } while (warm_ms < MIN_WARMUP_MS);
        CUDA_CHECK(cudaEventDestroy(ws)); CUDA_CHECK(cudaEventDestroy(we));
        cudaEvent_t s, e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        for (int i = 0; i < TIMED; i++) cg_matvec_3d_1<<<grid, block>>>(d_p, d_res_g, d_Ap);
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); ms /= TIMED;
        CUDA_CHECK(cudaMemcpy(h_result, d_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int i = 0; i < N; i++) max_err = fmax(max_err, fabs((double)h_result[i] - h_ref[i]));
        printf("%-10s %10.5f %10.3e %8s\n", "mdh", ms, max_err, max_err < 1e-3 ? "yes" : "no");
    }

    // ppcg (matrix-free)
    {
        dim3 block(4, 4, M), grid(1, 1);
        cudaEvent_t ws, we; CUDA_CHECK(cudaEventCreate(&ws)); CUDA_CHECK(cudaEventCreate(&we));
        CUDA_CHECK(cudaEventRecord(ws));
        float warm_ms;
        do {
            kernel0<<<grid, block>>>(d_Ap, d_p_padded);
            CUDA_CHECK(cudaEventRecord(we)); CUDA_CHECK(cudaEventSynchronize(we));
            CUDA_CHECK(cudaEventElapsedTime(&warm_ms, ws, we));
        } while (warm_ms < MIN_WARMUP_MS);
        CUDA_CHECK(cudaEventDestroy(ws)); CUDA_CHECK(cudaEventDestroy(we));
        cudaEvent_t s, e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        for (int i = 0; i < TIMED; i++) kernel0<<<grid, block>>>(d_Ap, d_p_padded);
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); ms /= TIMED;
        CUDA_CHECK(cudaMemcpy(h_result, d_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int i = 0; i < N; i++) max_err = fmax(max_err, fabs((double)h_result[i] - h_ref[i]));
        printf("%-10s %10.5f %10.3e %8s\n", "ppcg", ms, max_err, max_err < 1e-3 ? "yes" : "no");
    }

    // cusparse
    {
        cusparseHandle_t handle; CUSPARSE_CHECK(cusparseCreate(&handle));
        cusparseSpMatDescr_t matA;
        CUSPARSE_CHECK(cusparseCreateCsr(&matA, N, N, nnz, d_row_ptr, d_col_idx, d_val,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        cusparseDnVecDescr_t vecP, vecAp;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecP, N, d_p, CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecAp, N, d_Ap, CUDA_R_32F));
        const float alpha = 1.0f, beta = 0.0f;
        size_t bufSize = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecP, &beta, vecAp, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize));
        void* dBuffer; CUDA_CHECK(cudaMalloc(&dBuffer, bufSize));

        cudaEvent_t ws, we; CUDA_CHECK(cudaEventCreate(&ws)); CUDA_CHECK(cudaEventCreate(&we));
        CUDA_CHECK(cudaEventRecord(ws));
        float warm_ms;
        do {
            CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecP,
                &beta, vecAp, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
            CUDA_CHECK(cudaEventRecord(we)); CUDA_CHECK(cudaEventSynchronize(we));
            CUDA_CHECK(cudaEventElapsedTime(&warm_ms, ws, we));
        } while (warm_ms < MIN_WARMUP_MS);
        CUDA_CHECK(cudaEventDestroy(ws)); CUDA_CHECK(cudaEventDestroy(we));
        cudaEvent_t s, e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        for (int i = 0; i < TIMED; i++)
            CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecP,
                &beta, vecAp, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); ms /= TIMED;
        CUDA_CHECK(cudaMemcpy(h_result, d_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int i = 0; i < N; i++) max_err = fmax(max_err, fabs((double)h_result[i] - h_ref[i]));
        printf("%-10s %10.5f %10.3e %8s\n", "cusparse", ms, max_err, max_err < 1e-3 ? "yes" : "no");

        cusparseDestroySpMat(matA); cusparseDestroyDnVec(vecP); cusparseDestroyDnVec(vecAp);
        cusparseDestroy(handle); cudaFree(dBuffer);
    }

    return 0;
}
