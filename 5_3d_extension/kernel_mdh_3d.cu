// Build:
//   nvcc -O3 kernel_mdh_3d.cu cg_matvec_3d_1.cu -o kernel_mdh_3d \
//     -DTYPE_T=float -DTYPE_TS=float \
//     -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
//     -DG_CB_RES_DEST_LEVEL=2 \
//     -DG_CB_SIZE_L_1=16 -DG_CB_SIZE_L_2=16 -DG_CB_SIZE_L_3=16 \
//     -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=8 -DL_CB_SIZE_L_2=8 -DL_CB_SIZE_L_3=8 \
//     -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1 -DP_CB_SIZE_L_3=1 \
//     -DNUM_WG_L_1=2 -DNUM_WG_L_2=2 -DNUM_WG_L_3=2 \
//     -DNUM_WI_L_1=8 -DNUM_WI_L_2=8 -DNUM_WI_L_3=8 \
//     -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_L_3=0
// Run:
//   ./kernel_mdh_3d
//
// MDH-generated matrix-free 3D solver - the first working end-to-end proof
// that MDH extends cleanly to 3D with no changes to the generator itself.
// Direct 3D extension of
// ../2_mdh_sparse/kernel_mdh_sparse.cu; the generated kernel
// (cg_matvec_3d_1.cu) comes from
// mdh-cuda-stencil-kernels/kernels/cg_matvec_3d/spec/cg_matvec_3d.cpp,
// verified correct in isolation by test_mdh_matvec_3d.cu before being
// wired into this full solver.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#ifndef G_CB_SIZE_L_1
#error "G_CB_SIZE_L_1 (== interior grid side length) must be defined via -D"
#endif
#define FULLN (G_CB_SIZE_L_1 + 2)
#define TOL 1e-6
#define MAX_ITER 5000

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

static inline float u_exact(float x, float y, float z) {
    return 1.0f + x * x + y * y + z * z;
}

// Matrix-free MDH-generated matvec: Ap = A*p for the 7-point 3D Poisson
// stencil (diag=6, six face-neighbors -1, Dirichlet oob=0). A is never
// stored anywhere, not even as CSR.
extern __global__ void cg_matvec_3d_1(
    float const * const __restrict__ P,
    float       * const __restrict__ res_g,
    float       * const __restrict__ int_res
);

int main() {
    int fulln = FULLN;
    int m = fulln - 2;
    int N = m * m * m;
    float h = 1.0f / (fulln - 1);

    if (m != G_CB_SIZE_L_1) {
        printf("Grid/build mismatch: m=%d but compiled for G_CB_SIZE_L_1=%d.\n", m, G_CB_SIZE_L_1);
        return -1;
    }

    printf("N=%d (matrix-free: 0 bytes for A, vs %.2f MB dense)\n",
           N, (double)N * N * sizeof(float) / (1024.0 * 1024.0));

    float* b = (float*)malloc(N * sizeof(float));
    float* x = (float*)calloc(N, sizeof(float));
    float* r = (float*)malloc(N * sizeof(float));
    float* p = (float*)malloc(N * sizeof(float));
    float* Ap = (float*)malloc(N * sizeof(float));

    if (!b || !x || !r || !p || !Ap) {
        printf("Allocation failure\n");
        return -1;
    }

    for (int i = 0; i < m; i++) {
        float xh = (i + 1) * h;
        for (int j = 0; j < m; j++) {
            float yh = (j + 1) * h;
            for (int k = 0; k < m; k++) {
                float zh = (k + 1) * h;
                int idx = (i * m + j) * m + k;
                b[idx] = h * h * (-6.0f);
                if (i == 0)     b[idx] += u_exact(0.0f, yh, zh);
                if (i == m - 1) b[idx] += u_exact(1.0f, yh, zh);
                if (j == 0)     b[idx] += u_exact(xh, 0.0f, zh);
                if (j == m - 1) b[idx] += u_exact(xh, 1.0f, zh);
                if (k == 0)     b[idx] += u_exact(xh, yh, 0.0f);
                if (k == m - 1) b[idx] += u_exact(xh, yh, 1.0f);
            }
        }
    }
    printf("b[0]: %.3f\n", b[0]);

    float* dev_p; float* dev_Ap; float* dev_res_g;
    cudaCheckReturn(cudaMalloc((void**)&dev_p, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_Ap, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_res_g, N * sizeof(float))); // required by kernel signature, unused

    for (int i = 0; i < N; i++) { r[i] = b[i]; p[i] = r[i]; }

    float rs_old = 0.0f;
    for (int i = 0; i < N; i++) rs_old += r[i] * r[i];
    printf("rs_old: %.3f\n", rs_old);

    // OCL_DIM_L_1=2 -> block.z, OCL_DIM_L_2=1 -> block.y, OCL_DIM_L_3=0 -> block.x
    dim3 block(NUM_WI_L_3, NUM_WI_L_2, NUM_WI_L_1);
    dim3 grid(NUM_WG_L_3, NUM_WG_L_2, NUM_WG_L_1);

    clock_t start_time = clock();
    int iters = 0;

    for (int k = 0; k < MAX_ITER; k++) {
        iters = k + 1;
        cudaCheckReturn(cudaMemcpy(dev_p, p, N * sizeof(float), cudaMemcpyHostToDevice));

        cg_matvec_3d_1<<<grid, block>>>(dev_p, dev_res_g, dev_Ap);
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

    double max_err = 0.0;
    for (int i = 0; i < m; i++) {
        float xh = (i + 1) * h;
        for (int j = 0; j < m; j++) {
            float yh = (j + 1) * h;
            for (int k = 0; k < m; k++) {
                float zh = (k + 1) * h;
                int idx = (i * m + j) * m + k;
                double exact = u_exact(xh, yh, zh);
                double err = fabs((double)x[idx] - exact);
                if (err > max_err) max_err = err;
            }
        }
    }
    printf("Max abs error vs analytical solution: %.6e\n", max_err);
    printf("Time taken for main loop: %.3f ms\n", elapsed_time);

    cudaFree(dev_p); cudaFree(dev_Ap); cudaFree(dev_res_g);
    free(b); free(x); free(r); free(p); free(Ap);

    return 0;
}
