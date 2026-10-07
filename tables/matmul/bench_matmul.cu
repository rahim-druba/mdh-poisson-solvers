// 4-way dense matmul comparison: S = A*B, GEMM_N x GEMM_N x GEMM_N, row-major layout
// (A[i*GEMM_N+k], B[k*GEMM_N+j], S[i*GEMM_N+j]) matching PPCG's and MDH's conventions.
// GEMM_N is a -D compile-time macro so this single source builds all 4 sizes.
//
// The four "ways" (cuSPARSE doesn't do dense GEMM, so cuBLAS takes its slot
// here -- matches what the original article's Table 3 did):
//   naive   - hand-written one-thread-per-output kernel, no tiling
//   mdh     - MDH-generated GEMM kernel (reused from cg-ppcg-test/gemm_spec.cpp,
//             same source recompiled per size via -D tile macros, no regeneration)
//   ppcg    - PPCG-generated GEMM kernel (regenerated per size, ppcg picked its
//             own shared-memory tiling automatically)
//   cublas  - vendor library (cublasSgemm)
//
// Correctness: spot-check SAMPLES random output elements against an O(GEMM_N)
// per-element CPU dot product (full O(GEMM_N^3) CPU reference would be too slow
// at GEMM_N=4096). Methodology: cudaEvent, WARMUP + TIMED launches, averaged --
// counts kept low relative to the matvec table because a single GEMM call
// is itself O(GEMM_N^3) work, not microseconds.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "matmul_ppcg_src_kernel.hu"

#ifndef GEMM_N
#error "GEMM_N must be defined (-DN=..)"
#endif

static const int WARMUP = 2;
static const int TIMED  = 5;
static const int SAMPLES = 50;

#define CUDA_CHECK(e) do { \
    cudaError_t _err = (e); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(e) do { \
    cublasStatus_t _st = (e); \
    if (_st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_st, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

extern __global__ void gemm_1(
    float const * const __restrict__ A,
    float const * const __restrict__ B,
    float       * const __restrict__ res_g,
    float       * const __restrict__ int_res,
    float       * const __restrict__ S_orig);
// kernel0(A, B, S) declared via matmul_ppcg_src_kernel.hu

__global__ void naive_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                     float* __restrict__ S, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) sum += A[i * n + k] * B[k * n + j];
        S[i * n + j] = sum;
    }
}

double bytes_and_report(const char* name, float ms, double* out_gflops) {
    double flops = 2.0 * (double)GEMM_N * GEMM_N * GEMM_N;
    *out_gflops = flops / (ms * 1e-3) / 1e9;
    return ms;
}

int main() {
    printf("=== 4-way dense matmul comparison: S = A*B, GEMM_N=%d ===\n", GEMM_N);
    printf("Methodology: %d warmup + %d timed launches, cudaEvent, averaged.\n", WARMUP, TIMED);
    printf("Correctness: %d spot-checked output elements vs CPU dot product.\n\n", SAMPLES);

    size_t sz = (size_t)GEMM_N * GEMM_N;
    float* h_A = (float*)malloc(sz * sizeof(float));
    float* h_B = (float*)malloc(sz * sizeof(float));
    float* h_S = (float*)malloc(sz * sizeof(float));

    for (int i = 0; i < GEMM_N; i++)
        for (int k = 0; k < GEMM_N; k++)
            h_A[i * GEMM_N + k] = (float)((i + k) % 5) - 2.0f;
    for (int k = 0; k < GEMM_N; k++)
        for (int j = 0; j < GEMM_N; j++)
            h_B[k * GEMM_N + j] = (float)((k + 2 * j) % 5) - 2.0f;

    // spot-check positions + CPU reference for them
    srand(7);
    int* si = (int*)malloc(SAMPLES * sizeof(int));
    int* sj = (int*)malloc(SAMPLES * sizeof(int));
    double* sref = (double*)malloc(SAMPLES * sizeof(double));
    for (int s = 0; s < SAMPLES; s++) {
        si[s] = rand() % GEMM_N;
        sj[s] = rand() % GEMM_N;
        double acc = 0.0;
        for (int k = 0; k < GEMM_N; k++) acc += (double)h_A[si[s] * GEMM_N + k] * (double)h_B[k * GEMM_N + sj[s]];
        sref[s] = acc;
    }

    float *d_A, *d_B, *d_S;
    CUDA_CHECK(cudaMalloc(&d_A, sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, sz * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sz * sizeof(float), cudaMemcpyHostToDevice));

    auto check = [&](const char* name, float* d_result_flat) {
        CUDA_CHECK(cudaMemcpy(h_S, d_result_flat, sz * sizeof(float), cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int s = 0; s < SAMPLES; s++) {
            double got = h_S[si[s] * GEMM_N + sj[s]];
            double err = fabs(got - sref[s]);
            if (err > max_err) max_err = err;
        }
        return max_err;
    };

    printf("%-10s %10s %12s %14s %10s\n", "method", "avg ms", "max error", "GFLOP/s", "correct");
    printf("------------------------------------------------------------------\n");

    // -- naive --
    {
        dim3 block(16, 16), grid((GEMM_N + 15) / 16, (GEMM_N + 15) / 16);
        for (int i = 0; i < WARMUP; i++) naive_matmul_kernel<<<grid, block>>>(d_A, d_B, d_S, GEMM_N);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t s0, s1; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&s1));
        CUDA_CHECK(cudaEventRecord(s0));
        for (int i = 0; i < TIMED; i++) naive_matmul_kernel<<<grid, block>>>(d_A, d_B, d_S, GEMM_N);
        CUDA_CHECK(cudaEventRecord(s1)); CUDA_CHECK(cudaEventSynchronize(s1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1)); ms /= TIMED;
        double gflops; bytes_and_report("naive", ms, &gflops);
        double max_err = check("naive", d_S);
        printf("%-10s %10.3f %12.3e %14.2f %10s\n", "naive", ms, max_err, gflops, max_err < 1.0 ? "yes" : "NO");
    }

    // -- mdh --
    {
        float *d_res_g, *d_int_res, *d_S_orig;
        CUDA_CHECK(cudaMalloc(&d_res_g, sz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_int_res, sz * sizeof(float) * NUM_WG_R_1));
        CUDA_CHECK(cudaMalloc(&d_S_orig, sz * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_int_res, 0, sz * sizeof(float) * NUM_WG_R_1));

        dim3 block(NUM_WI_R_1, NUM_WI_L_2, NUM_WI_L_1);
        dim3 grid(NUM_WG_R_1, NUM_WG_L_2, NUM_WG_L_1);

        for (int i = 0; i < WARMUP; i++) gemm_1<<<grid, block>>>(d_A, d_B, d_res_g, d_int_res, d_S_orig);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t s0, s1; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&s1));
        CUDA_CHECK(cudaEventRecord(s0));
        for (int i = 0; i < TIMED; i++) gemm_1<<<grid, block>>>(d_A, d_B, d_res_g, d_int_res, d_S_orig);
        CUDA_CHECK(cudaEventRecord(s1)); CUDA_CHECK(cudaEventSynchronize(s1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1)); ms /= TIMED;
        double gflops; bytes_and_report("mdh", ms, &gflops);
        double max_err = check("mdh", d_int_res); // NUM_WG_R_1==1 -> int_res holds full result
        printf("%-10s %10.3f %12.3e %14.2f %10s\n", "mdh", ms, max_err, gflops, max_err < 1.0 ? "yes" : "NO");

        cudaFree(d_res_g); cudaFree(d_int_res); cudaFree(d_S_orig);
    }

    // -- ppcg --
    {
        dim3 block(PPCG_BLOCK_X, PPCG_BLOCK_Y);
        dim3 grid(PPCG_GRID_X, PPCG_GRID_Y);
        for (int i = 0; i < WARMUP; i++) kernel0<<<grid, block>>>(d_A, d_B, d_S);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t s0, s1; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&s1));
        CUDA_CHECK(cudaEventRecord(s0));
        for (int i = 0; i < TIMED; i++) kernel0<<<grid, block>>>(d_A, d_B, d_S);
        CUDA_CHECK(cudaEventRecord(s1)); CUDA_CHECK(cudaEventSynchronize(s1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1)); ms /= TIMED;
        double gflops; bytes_and_report("ppcg", ms, &gflops);
        double max_err = check("ppcg", d_S);
        printf("%-10s %10.3f %12.3e %14.2f %10s\n", "ppcg", ms, max_err, gflops, max_err < 1.0 ? "yes" : "NO");
    }

    // -- cublas --
    {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        const float alpha = 1.0f, beta = 0.0f;

        for (int i = 0; i < WARMUP; i++)
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, GEMM_N, GEMM_N, GEMM_N, &alpha, d_B, GEMM_N, d_A, GEMM_N, &beta, d_S, GEMM_N));
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t s0, s1; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&s1));
        CUDA_CHECK(cudaEventRecord(s0));
        for (int i = 0; i < TIMED; i++)
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, GEMM_N, GEMM_N, GEMM_N, &alpha, d_B, GEMM_N, d_A, GEMM_N, &beta, d_S, GEMM_N));
        CUDA_CHECK(cudaEventRecord(s1)); CUDA_CHECK(cudaEventSynchronize(s1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1)); ms /= TIMED;
        double gflops; bytes_and_report("cublas", ms, &gflops);
        double max_err = check("cublas", d_S);
        printf("%-10s %10.3f %12.3e %14.2f %10s\n", "cublas", ms, max_err, gflops, max_err < 1.0 ? "yes" : "NO");
        cublasDestroy(handle);
    }

    return 0;
}
