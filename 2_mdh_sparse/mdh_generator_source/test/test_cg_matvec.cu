// Correctness test for the MDH-generated matrix-free CG matvec kernel.
// Compares GPU output against a serial CPU reference of Ap = A*p
// (5-point Poisson stencil: diagonal 4, neighbors -1, Dirichlet oob=0).
//
// Build:
//   nvcc test_cg_matvec.cu cg_matvec_1.cu -o test_cg_matvec       \
//     -DTYPE_T=float -DTYPE_TS=float                              \
//     -DCACHE_L_CB=0 -DCACHE_P_CB=0                                \
//     -DG_CB_RES_DEST_LEVEL=2                                       \
//     -DG_CB_SIZE_L_1=64 -DG_CB_SIZE_L_2=64                        \
//     -DL_CB_RES_DEST_LEVEL=1                                       \
//     -DL_CB_SIZE_L_1=16 -DL_CB_SIZE_L_2=16                        \
//     -DNUM_WG_L_1=4   -DNUM_WG_L_2=4                              \
//     -DNUM_WI_L_1=16  -DNUM_WI_L_2=16                             \
//     -DOCL_DIM_L_1=1  -DOCL_DIM_L_2=0                             \
//     -DP_CB_RES_DEST_LEVEL=0                                       \
//     -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

static const int NI = 64;   // interior grid side, matches kernel_sparse.cu m=64 (fulln=66)

extern __global__ void cg_matvec_1(
    float const * const __restrict__ P,
    float       * const __restrict__ res_g,
    float       * const __restrict__ int_res
);

#define CUDA_CHECK(e) do {                                                   \
    cudaError_t _err = (e);                                                  \
    if (_err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error '%s' at %s:%d\n",                       \
                cudaGetErrorString(_err), __FILE__, __LINE__);               \
        exit(1);                                                             \
    }                                                                        \
} while(0)

int main() {
    const int sz = NI * NI;

    printf("=== MDH CG Matvec (matrix-free 5-point stencil) Test ===\n");
    printf("Interior grid: %d x %d = %d points\n", NI, NI, sz);

    float* h_P      = new float[sz];
    float* h_result = new float[sz]();
    float* h_ref    = new float[sz]();

    srand(42);
    for (int i = 0; i < sz; i++) h_P[i] = (float)(rand() % 100) / 10.0f - 5.0f;

    // serial reference: Ap[i,j] = 4*P[i,j] - top - bottom - left - right, oob=0
    for (int k = 0; k < NI; k++) {
        for (int l = 0; l < NI; l++) {
            float top    = (k > 0)      ? h_P[(k-1)*NI + l] : 0.0f;
            float bottom = (k < NI - 1) ? h_P[(k+1)*NI + l] : 0.0f;
            float left   = (l > 0)      ? h_P[k*NI + (l-1)] : 0.0f;
            float right  = (l < NI - 1) ? h_P[k*NI + (l+1)] : 0.0f;
            h_ref[k*NI + l] = 4.0f * h_P[k*NI + l] - top - bottom - left - right;
        }
    }

    float *d_P, *d_res_g, *d_int_res;
    CUDA_CHECK(cudaMalloc(&d_P,       sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_g,   sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_int_res, sz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_P, h_P, sz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_res_g,   0, sz * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_int_res, 0, sz * sizeof(float)));

    dim3 block(NUM_WI_L_2, NUM_WI_L_1, 1);
    dim3 grid(NUM_WG_L_2, NUM_WG_L_1, 1);

    printf("Grid: (%d,%d)  Block: (%d,%d)\n", grid.x, grid.y, block.x, block.y);

    cudaEvent_t t_start, t_stop;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_stop));

    CUDA_CHECK(cudaEventRecord(t_start));
    cg_matvec_1<<<grid, block>>>(d_P, d_res_g, d_int_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t_stop));
    CUDA_CHECK(cudaEventSynchronize(t_stop));

    CUDA_CHECK(cudaMemcpy(h_result, d_int_res, sz * sizeof(float), cudaMemcpyDeviceToHost));

    double max_error = 0.0;
    int mismatches = 0;
    for (int idx = 0; idx < sz; idx++) {
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
    printf("Elements checked : %d\n", sz);
    printf("Max error vs CPU : %.3e\n", max_error);
    printf("Mismatches       : %d\n", mismatches);
    printf("GPU time         : %.4f ms\n", gpu_ms);
    printf(mismatches == 0 ? "\nMDH matrix-free CG matvec is CORRECT!\n"
                            : "\nMDH matrix-free CG matvec FAILED!\n");

    return mismatches == 0 ? 0 : 1;
}
