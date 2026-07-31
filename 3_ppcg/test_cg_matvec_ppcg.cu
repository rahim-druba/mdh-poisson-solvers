// Correctness test for the PPCG-generated matrix-free CG matvec kernel
// (kernel0, from cg_matvec_ppcg_src_kernel.cu). Same reference check as
// 2_mdh_sparse/mdh_generator_source/test/test_cg_matvec.cu, so the two
// generators can be compared apples-to-apples.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cg_matvec_ppcg_src_kernel.hu"

static const int M = 64;   // interior grid side
static const int PAD = M + 2;

#define CUDA_CHECK(e) do {                                                   \
    cudaError_t _err = (e);                                                  \
    if (_err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error '%s' at %s:%d\n",                       \
                cudaGetErrorString(_err), __FILE__, __LINE__);               \
        exit(1);                                                             \
    }                                                                        \
} while(0)

int main() {
    const int sz_p = PAD * PAD;
    const int sz_ap = M * M;

    printf("=== PPCG CG Matvec (matrix-free 5-point stencil) Test ===\n");
    printf("Interior grid: %d x %d = %d points\n", M, M, sz_ap);

    float* h_P      = new float[sz_p]();
    float* h_result = new float[sz_ap]();
    float* h_ref    = new float[sz_ap]();

    srand(42);
    for (int i = 1; i <= M; i++)
        for (int j = 1; j <= M; j++)
            h_P[i * PAD + j] = (float)(rand() % 100) / 10.0f - 5.0f;

    // serial reference: same operator as kernel_sparse.cu / cg_matvec_1
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            float center = h_P[(i + 1) * PAD + (j + 1)];
            float top    = h_P[i * PAD + (j + 1)];
            float bottom = h_P[(i + 2) * PAD + (j + 1)];
            float left   = h_P[(i + 1) * PAD + j];
            float right  = h_P[(i + 1) * PAD + (j + 2)];
            h_ref[i * M + j] = 4.0f * center - top - bottom - left - right;
        }
    }

    float *d_P, *d_Ap;
    CUDA_CHECK(cudaMalloc(&d_P,  sz_p  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ap, sz_ap * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_P, h_P, sz_p * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 32);
    dim3 grid(2, 2);

    printf("Grid: (%d,%d)  Block: (%d,%d)\n", grid.x, grid.y, block.x, block.y);

    cudaEvent_t t_start, t_stop;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_stop));

    CUDA_CHECK(cudaEventRecord(t_start));
    kernel0<<<grid, block>>>(d_Ap, d_P);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t_stop));
    CUDA_CHECK(cudaEventSynchronize(t_stop));

    CUDA_CHECK(cudaMemcpy(h_result, d_Ap, sz_ap * sizeof(float), cudaMemcpyDeviceToHost));

    double max_error = 0.0;
    int mismatches = 0;
    for (int idx = 0; idx < sz_ap; idx++) {
        double err = fabs((double)h_result[idx] - (double)h_ref[idx]);
        if (err > max_error) max_error = err;
        if (err > 1e-4) {
            if (mismatches < 5)
                printf("  MISMATCH idx=%d  gpu=%.6f  ref=%.6f  diff=%.3e\n",
                       idx, h_result[idx], h_ref[idx], err);
            mismatches++;
        }
    }

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t_start, t_stop));

    printf("\n--- Results ---\n");
    printf("Elements checked : %d\n", sz_ap);
    printf("Max error vs CPU : %.3e\n", max_error);
    printf("Mismatches       : %d\n", mismatches);
    printf("GPU time         : %.4f ms\n", gpu_ms);
    printf(mismatches == 0 ? "\nPPCG matrix-free CG matvec is CORRECT!\n"
                            : "\nPPCG matrix-free CG matvec FAILED!\n");

    return mismatches == 0 ? 0 : 1;
}
