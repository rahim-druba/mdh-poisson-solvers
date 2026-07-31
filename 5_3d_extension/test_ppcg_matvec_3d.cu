// Verifies the PPCG-generated 3D matvec kernel (kernel0, in
// cg_matvec_3d_ppcg_src_kernel.cu) against the CPU reference, at the same
// M=16 (interior 16x16x16 = 4096 unknowns) grid the kernel was generated
// for -- PPCG bakes sizes in at generation time, same as the 2D pipeline.
//
// Build:
//   nvcc -O3 test_ppcg_matvec_3d.cu cg_matvec_3d_ppcg_src_kernel.cu \
//     -o test_ppcg_matvec_3d
// Run:
//   ./test_ppcg_matvec_3d

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cpu_reference_matvec_3d.h"
#include "cg_matvec_3d_ppcg_src_kernel.hu"

#define M 16
#define MP2 (M + 2)

#define cudaCheckReturn(ret) \
  do { \
    cudaError_t e = (ret); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s (at %s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

int main() {
    const int N = M * M * M;
    const int NP = MP2 * MP2 * MP2;

    float* p = (float*)malloc(N * sizeof(float));
    float* Ap_cpu = (float*)malloc(N * sizeof(float));
    float* Ap_gpu = (float*)malloc(N * sizeof(float));
    float* P_padded = (float*)calloc(NP, sizeof(float)); // zero halo, same convention as the generated host code

    srand(0);
    for (int i = 0; i < N; ++i) p[i] = (float)(rand() % 100) / 10.0f - 5.0f;

    cpu_matvec_3d(M, p, Ap_cpu);

    // Build the zero-padded P array PPCG's kernel expects (halo of 1 on all 6 faces)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j)
            for (int k = 0; k < M; ++k)
                P_padded[((i + 1) * MP2 + (j + 1)) * MP2 + (k + 1)] = p[(i * M + j) * M + k];

    float *dev_Ap, *dev_P;
    cudaCheckReturn(cudaMalloc((void**)&dev_Ap, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_P, NP * sizeof(float)));
    cudaCheckReturn(cudaMemcpy(dev_P, P_padded, NP * sizeof(float), cudaMemcpyHostToDevice));

    dim3 k0_dimBlock(4, 4, 16);
    dim3 k0_dimGrid(1, 1);
    kernel0<<<k0_dimGrid, k0_dimBlock>>>(dev_Ap, dev_P);
    cudaCheckReturn(cudaGetLastError());
    cudaCheckReturn(cudaDeviceSynchronize());

    cudaCheckReturn(cudaMemcpy(Ap_gpu, dev_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));

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

    cudaFree(dev_Ap); cudaFree(dev_P);
    free(p); free(Ap_cpu); free(Ap_gpu); free(P_padded);
    return mismatches == 0 ? 0 : 1;
}
