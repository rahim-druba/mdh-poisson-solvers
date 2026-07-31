// Build (M = interior grid side length -- must match the size directory's
// PPCG-generated kernel being linked in, e.g. for M=16:
//   nvcc -O3 -DM=16 -I sizes/16 kernel_ppcg_3d.cu sizes/16/cg_matvec_3d_ppcg_src_kernel.cu \
//     -o kernel_ppcg_3d_M16
// Run:
//   ./kernel_ppcg_3d_M16
//
// PPCG-generated matrix-free 3D solver -- direct 3D extension of
// ../3_ppcg/kernel_ppcg_sparse.cu. kernel0 (PPCG-generated, from
// sizes/<M>/cg_matvec_3d_ppcg_src.c) is baked for an MxMxM grid with a
// 1-cell zero halo ((M+2)^3 padded input); PPCG picks block(4,4,M),
// grid(1,1) consistently across every size generated so far.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cg_matvec_3d_ppcg_src_kernel.hu"

#ifndef M
#error "M must be defined (-DM=.. interior grid side length)"
#endif
#define FULLN (M + 2)
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

int main() {
    int fulln = FULLN;
    int m = fulln - 2;
    int N = m * m * m;
    int pad = m + 2;
    float h = 1.0f / (fulln - 1);

    if (m != M) {
        printf("Grid/build mismatch: m=%d but compiled for M=%d.\n", m, M);
        return -1;
    }

    printf("N=%d (matrix-free: 0 bytes for A, vs %.2f MB dense)\n",
           N, (double)N * N * sizeof(float) / (1024.0 * 1024.0));

    float* b = (float*)malloc(N * sizeof(float));
    float* x = (float*)calloc(N, sizeof(float));
    float* r = (float*)malloc(N * sizeof(float));
    float* p = (float*)malloc(N * sizeof(float));
    float* Ap = (float*)malloc(N * sizeof(float));
    float* p_padded = (float*)calloc(pad * pad * pad, sizeof(float)); // halo stays 0 forever

    if (!b || !x || !r || !p || !Ap || !p_padded) {
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

    float* dev_p_padded; float* dev_Ap;
    cudaCheckReturn(cudaMalloc((void**)&dev_p_padded, pad * pad * pad * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_Ap, N * sizeof(float)));
    cudaCheckReturn(cudaMemset(dev_p_padded, 0, pad * pad * pad * sizeof(float)));

    for (int i = 0; i < N; i++) { r[i] = b[i]; p[i] = r[i]; }

    float rs_old = 0.0f;
    for (int i = 0; i < N; i++) rs_old += r[i] * r[i];
    printf("rs_old: %.3f\n", rs_old);

    dim3 block(4, 4, M);  // matches PPCG-generated launch config (k0_dimBlock) at every size so far
    dim3 grid(1, 1);

    clock_t start_time = clock();
    int iters = 0;

    for (int k = 0; k < MAX_ITER; k++) {
        iters = k + 1;

        // copy p into the interior of the zero-padded buffer, plane by plane, row by row
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                memcpy(p_padded + ((i + 1) * pad + (j + 1)) * pad + 1,
                       p + (i * m + j) * m,
                       m * sizeof(float));
            }
        }
        cudaCheckReturn(cudaMemcpy(dev_p_padded, p_padded, pad * pad * pad * sizeof(float), cudaMemcpyHostToDevice));

        kernel0<<<grid, block>>>(dev_Ap, dev_p_padded);
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

    cudaFree(dev_p_padded); cudaFree(dev_Ap);
    free(b); free(x); free(r); free(p); free(Ap); free(p_padded);

    return 0;
}
