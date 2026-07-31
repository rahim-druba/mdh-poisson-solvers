// 4-way matvec comparison: Ap = A*p for the CG solver's 2D 5-point Poisson
// stencil (diag=4, neighbors=-1), on a ROWS x COLS interior grid (N = ROWS*COLS).
// ROWS, COLS, PPCG_BLOCK_*, PPCG_GRID_* are passed as -D compile-time macros
// so this single source builds all 4 sizes (512/1024/2048/4096) -- see
// run_matvec_sweep.sh for the exact per-size build commands.
//
// Same four "ways" as the N=4096 version in ../../comparison/bench_matvec_4way.cu:
//   sparse    - hand-written CSR-scalar kernel
//   mdh       - MDH-generated matrix-free stencil kernel (same source, recompiled per size)
//   ppcg      - PPCG-generated matrix-free stencil kernel (regenerated per size)
//   cusparse  - vendor library, true CSR
//
// Methodology: cudaEvent timing, 20 warmup + 200 timed launches, averaged.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "matvec_ppcg_src_kernel.hu"

#ifndef ROWS
#error "ROWS must be defined (-DROWS=..)"
#endif
#ifndef COLS
#error "COLS must be defined (-DCOLS=..)"
#endif

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

static const int R = ROWS;
static const int C = COLS;
static const int N = R * C;
static const int PAD_R = R + 2;
static const int PAD_C = C + 2;

static const int TIMED = 200;
// Time-based warmup instead of a fixed iteration count -- see the dense
// matvec table's writeup for why (laptop GPU power states need a minimum
// sustained-load duration to stabilize, not a minimum iteration count).
static const float MIN_WARMUP_MS = 300.0f;

extern __global__ void cg_matvec_1(
    float const * const __restrict__ P,
    float       * const __restrict__ res_g,
    float       * const __restrict__ int_res
);
// kernel0(Ap, P) declared via matvec_ppcg_src_kernel.hu

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
    printf("=== 4-way matvec comparison: Ap = A*p, N=%d (%dx%d grid) ===\n", N, R, C);
    printf("Methodology: >=%.0fms time-based warmup + %d timed launches, cudaEvent, averaged.\n\n", MIN_WARMUP_MS, TIMED);

    float* h_p = (float*)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) h_p[i] = (float)(rand() % 100) / 10.0f - 5.0f;

    float* h_ref = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            float center = h_p[i * C + j];
            float top    = (i > 0)     ? h_p[(i - 1) * C + j] : 0.0f;
            float bottom = (i < R - 1) ? h_p[(i + 1) * C + j] : 0.0f;
            float left   = (j > 0)     ? h_p[i * C + (j - 1)] : 0.0f;
            float right  = (j < C - 1) ? h_p[i * C + (j + 1)] : 0.0f;
            h_ref[i * C + j] = 4.0f * center - top - bottom - left - right;
        }
    }

    int *row_ptr, *col_idx; float *val;
    int nnz = build_poisson_csr(R, C, &row_ptr, &col_idx, &val);
    int *d_row_ptr, *d_col_idx; float *d_val;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, nnz * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, val, nnz * sizeof(float), cudaMemcpyHostToDevice));

    float* h_p_padded = (float*)calloc(PAD_R * PAD_C, sizeof(float));
    for (int i = 0; i < R; i++)
        memcpy(h_p_padded + (i + 1) * PAD_C + 1, h_p + i * C, C * sizeof(float));

    float *d_p, *d_Ap, *d_p_padded, *d_res_g;
    CUDA_CHECK(cudaMalloc(&d_p, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ap, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_padded, PAD_R * PAD_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_g, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_p, h_p, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_padded, h_p_padded, PAD_R * PAD_C * sizeof(float), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
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
    void* d_buf; CUDA_CHECK(cudaMalloc(&d_buf, bufSize));

    dim3 naive_block(256), naive_grid((N + 255) / 256);
    dim3 mdh_block(NUM_WI_L_2, NUM_WI_L_1), mdh_grid(NUM_WG_L_2, NUM_WG_L_1);
    dim3 ppcg_block(PPCG_BLOCK_X, PPCG_BLOCK_Y), ppcg_grid(PPCG_GRID_X, PPCG_GRID_Y);

    float* h_result = (float*)malloc(N * sizeof(float));

    printf("%-10s %10s %12s %10s\n", "method", "avg ms", "max error", "correct");
    printf("--------------------------------------------------\n");

    { // sparse (naive CSR)
        cudaEvent_t ws, we; CUDA_CHECK(cudaEventCreate(&ws)); CUDA_CHECK(cudaEventCreate(&we));
        CUDA_CHECK(cudaEventRecord(ws));
        float warm_ms = 0.0f;
        do {
            spmv_csr_kernel<<<naive_grid, naive_block>>>(N, d_row_ptr, d_col_idx, d_val, d_p, d_Ap);
            CUDA_CHECK(cudaEventRecord(we)); CUDA_CHECK(cudaEventSynchronize(we));
            CUDA_CHECK(cudaEventElapsedTime(&warm_ms, ws, we));
        } while (warm_ms < MIN_WARMUP_MS);
        CUDA_CHECK(cudaEventDestroy(ws)); CUDA_CHECK(cudaEventDestroy(we));
        cudaEvent_t s, e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        for (int i = 0; i < TIMED; i++) spmv_csr_kernel<<<naive_grid, naive_block>>>(N, d_row_ptr, d_col_idx, d_val, d_p, d_Ap);
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); ms /= TIMED;
        CUDA_CHECK(cudaMemcpy(h_result, d_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int i = 0; i < N; i++) max_err = fmax(max_err, fabs((double)h_result[i] - h_ref[i]));
        printf("%-10s %10.5f %12.3e %10s\n", "sparse", ms, max_err, max_err < 1e-3 ? "yes" : "NO");
    }

    { // mdh
        cudaEvent_t ws, we; CUDA_CHECK(cudaEventCreate(&ws)); CUDA_CHECK(cudaEventCreate(&we));
        CUDA_CHECK(cudaEventRecord(ws));
        float warm_ms = 0.0f;
        do {
            cg_matvec_1<<<mdh_grid, mdh_block>>>(d_p, d_res_g, d_Ap);
            CUDA_CHECK(cudaEventRecord(we)); CUDA_CHECK(cudaEventSynchronize(we));
            CUDA_CHECK(cudaEventElapsedTime(&warm_ms, ws, we));
        } while (warm_ms < MIN_WARMUP_MS);
        CUDA_CHECK(cudaEventDestroy(ws)); CUDA_CHECK(cudaEventDestroy(we));
        cudaEvent_t s, e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        for (int i = 0; i < TIMED; i++) cg_matvec_1<<<mdh_grid, mdh_block>>>(d_p, d_res_g, d_Ap);
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); ms /= TIMED;
        CUDA_CHECK(cudaMemcpy(h_result, d_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int i = 0; i < N; i++) max_err = fmax(max_err, fabs((double)h_result[i] - h_ref[i]));
        printf("%-10s %10.5f %12.3e %10s\n", "mdh", ms, max_err, max_err < 1e-3 ? "yes" : "NO");
    }

    { // ppcg
        cudaEvent_t ws, we; CUDA_CHECK(cudaEventCreate(&ws)); CUDA_CHECK(cudaEventCreate(&we));
        CUDA_CHECK(cudaEventRecord(ws));
        float warm_ms = 0.0f;
        do {
            kernel0<<<ppcg_grid, ppcg_block>>>(d_Ap, d_p_padded);
            CUDA_CHECK(cudaEventRecord(we)); CUDA_CHECK(cudaEventSynchronize(we));
            CUDA_CHECK(cudaEventElapsedTime(&warm_ms, ws, we));
        } while (warm_ms < MIN_WARMUP_MS);
        CUDA_CHECK(cudaEventDestroy(ws)); CUDA_CHECK(cudaEventDestroy(we));
        cudaEvent_t s, e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        for (int i = 0; i < TIMED; i++) kernel0<<<ppcg_grid, ppcg_block>>>(d_Ap, d_p_padded);
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); ms /= TIMED;
        CUDA_CHECK(cudaMemcpy(h_result, d_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int i = 0; i < N; i++) max_err = fmax(max_err, fabs((double)h_result[i] - h_ref[i]));
        printf("%-10s %10.5f %12.3e %10s\n", "ppcg", ms, max_err, max_err < 1e-3 ? "yes" : "NO");
    }

    { // cusparse
        cudaEvent_t ws, we; CUDA_CHECK(cudaEventCreate(&ws)); CUDA_CHECK(cudaEventCreate(&we));
        CUDA_CHECK(cudaEventRecord(ws));
        float warm_ms = 0.0f;
        do {
            CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecP, &beta, vecAp, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf));
            CUDA_CHECK(cudaEventRecord(we)); CUDA_CHECK(cudaEventSynchronize(we));
            CUDA_CHECK(cudaEventElapsedTime(&warm_ms, ws, we));
        } while (warm_ms < MIN_WARMUP_MS);
        CUDA_CHECK(cudaEventDestroy(ws)); CUDA_CHECK(cudaEventDestroy(we));
        cudaEvent_t s, e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        for (int i = 0; i < TIMED; i++) CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecP, &beta, vecAp, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf));
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); ms /= TIMED;
        CUDA_CHECK(cudaMemcpy(h_result, d_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int i = 0; i < N; i++) max_err = fmax(max_err, fabs((double)h_result[i] - h_ref[i]));
        printf("%-10s %10.5f %12.3e %10s\n", "cusparse", ms, max_err, max_err < 1e-3 ? "yes" : "NO");
    }

    return 0;
}
