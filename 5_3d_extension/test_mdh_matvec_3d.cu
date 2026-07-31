// Smoke test: MDH-generated 3D matrix-free matvec (cg_matvec_3d_1) vs the
// CPU reference (cpu_reference_matvec_3d.h), at a tiny interior grid so a
// dimension-handling bug is cheap to catch before scaling up. This is the
// first-ever 3D MDH kernel in this framework -- no prior art to copy from.
//
// Build (tiny grid, M=4 interior per dim, one block covers the whole
// domain so NUM_WG_L_i=1, NUM_WI_L_i=4):
//   nvcc -O3 test_mdh_matvec_3d.cu cg_matvec_3d_1.cu -o test_mdh_matvec_3d \
//     -DTYPE_T=float -DTYPE_TS=float \
//     -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
//     -DG_CB_RES_DEST_LEVEL=2 \
//     -DG_CB_SIZE_L_1=4 -DG_CB_SIZE_L_2=4 -DG_CB_SIZE_L_3=4 \
//     -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=4 -DL_CB_SIZE_L_2=4 -DL_CB_SIZE_L_3=4 \
//     -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1 -DP_CB_SIZE_L_3=1 \
//     -DNUM_WG_L_1=1 -DNUM_WG_L_2=1 -DNUM_WG_L_3=1 \
//     -DNUM_WI_L_1=4 -DNUM_WI_L_2=4 -DNUM_WI_L_3=4 \
//     -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_L_3=0
// Run:
//   ./test_mdh_matvec_3d

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cpu_reference_matvec_3d.h"

#ifndef M
#define M 4
#endif

#define cudaCheckReturn(ret) \
  do { \
    cudaError_t e = (ret); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s (at %s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

extern __global__ void cg_matvec_3d_1(
    float const * const __restrict__ P,
    float       * const __restrict__ res_g,
    float       * const __restrict__ int_res
);

int main() {
    const int N = M * M * M;

    float* p = (float*)malloc(N * sizeof(float));
    float* Ap_cpu = (float*)malloc(N * sizeof(float));
    float* Ap_gpu = (float*)malloc(N * sizeof(float));

    srand(0);
    for (int i = 0; i < N; ++i) p[i] = (float)(rand() % 100) / 10.0f - 5.0f;

    cpu_matvec_3d(M, p, Ap_cpu);

    float *dev_p, *dev_res_g, *dev_int_res;
    cudaCheckReturn(cudaMalloc((void**)&dev_p, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_res_g, N * sizeof(float))); // required by kernel signature, unused here
    cudaCheckReturn(cudaMalloc((void**)&dev_int_res, N * sizeof(float)));
    cudaCheckReturn(cudaMemcpy(dev_p, p, N * sizeof(float), cudaMemcpyHostToDevice));

    // OCL_DIM_L_1=2 -> block.z, OCL_DIM_L_2=1 -> block.y, OCL_DIM_L_3=0 -> block.x
    dim3 block(NUM_WI_L_3, NUM_WI_L_2, NUM_WI_L_1);
    dim3 grid(NUM_WG_L_3, NUM_WG_L_2, NUM_WG_L_1);

    cg_matvec_3d_1<<<grid, block>>>(dev_p, dev_res_g, dev_int_res);
    cudaCheckReturn(cudaGetLastError());
    cudaCheckReturn(cudaDeviceSynchronize());

    cudaCheckReturn(cudaMemcpy(Ap_gpu, dev_int_res, N * sizeof(float), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = fabsf(Ap_gpu[i] - Ap_cpu[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-4f) {
            if (mismatches < 10) {
                printf("MISMATCH idx=%d gpu=%.6f cpu=%.6f diff=%.3e\n", i, Ap_gpu[i], Ap_cpu[i], err);
            }
            mismatches++;
        }
    }

    printf("N=%d (M=%d), mismatches=%d, max_err=%.3e\n", N, M, mismatches, max_err);
    printf(mismatches == 0 ? "PASS\n" : "FAIL\n");

    cudaFree(dev_p); cudaFree(dev_res_g); cudaFree(dev_int_res);
    free(p); free(Ap_cpu); free(Ap_gpu);
    return mismatches == 0 ? 0 : 1;
}
