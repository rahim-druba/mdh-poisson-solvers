#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cg_matvec_ppcg_src_kernel.hu"

// Same test problem as ../1_sparse_rewrite/kernel_sparse.cu and
// ../2_mdh_sparse/kernel_mdh_sparse.cu: fulln=66 -> interior grid 64x64 -> N=4096.
// kernel0 (PPCG-generated) is baked for a 64x64 grid with a 1-cell zero halo
// (66x66 padded input) -- see cg_matvec_ppcg_src.c.
#define FULLN 66
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

int main() {
    int fulln = FULLN;
    int m = fulln - 2;          // interior grid side length (must be 64 -- kernel0 is baked for it)
    int N = m * m;
    int pad = m + 2;             // padded side for the halo kernel0 expects
    float xh, yh, h;
    h = 1.0f / (fulln - 1);

    if (m != 64) {
        printf("This build's PPCG kernel0 is baked for a 64x64 interior grid.\n");
        return -1;
    }

    printf("N=%d (matrix-free: 0 bytes for A, vs %.2f MB dense)\n",
           N, (double)N * N * sizeof(float) / (1024.0 * 1024.0));

    // ---- Boundary values W and right-hand side b (same construction as the other two) ----
    float* W = (float*)malloc(fulln * fulln * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));
    float* x = (float*)calloc(N, sizeof(float));
    float* r = (float*)malloc(N * sizeof(float));
    float* p = (float*)malloc(N * sizeof(float));
    float* Ap = (float*)malloc(N * sizeof(float));
    float* p_padded = (float*)calloc(pad * pad, sizeof(float)); // halo stays 0 forever

    if (!W || !b || !x || !r || !p || !Ap || !p_padded) {
        printf("Allocation failure\n");
        return -1;
    }

    for (int i = 0; i < fulln; i++) {
        xh = i * h;
        W[0 * fulln + i] = 1 + xh * xh;
        W[i * fulln + 0] = 1 + xh * xh;
        W[(fulln - 1) * fulln + i] = 1 + 1 + xh * xh;
        W[i * fulln + (fulln - 1)] = 1 + 1 + xh * xh;
    }
    for (int i = 0; i < m; i++) {
        xh = (i + 1) * h;
        for (int j = 0; j < m; j++) {
            yh = (j + 1) * h;
            b[i * m + j] = h * h * (-4);
            if (i == 0) { b[i * m + j] += W[0 * fulln + j + 1]; }
            if (i == m - 1) { b[i * m + j] += W[(i + 2) * fulln + j + 1]; }
            if (j == 0) { b[i * m + j] += W[(i + 1) * fulln]; }
            if (j == m - 1) { b[i * m + j] += W[(i + 1) * fulln + fulln - 1]; }
        }
    }
    printf("b[0]: %.3f\n", b[0]);

    float* dev_p_padded; float* dev_Ap;
    cudaCheckReturn(cudaMalloc((void**)&dev_p_padded, pad * pad * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_Ap, N * sizeof(float)));
    // zero the padded buffer once -- the halo (border) is never touched again
    cudaCheckReturn(cudaMemset(dev_p_padded, 0, pad * pad * sizeof(float)));

    // ---- CG initialization: r = b - A*x, x = 0 so r = b ----
    for (int i = 0; i < N; i++) {
        r[i] = b[i];
        p[i] = r[i];
    }

    float rs_old = 0.0f;
    for (int i = 0; i < N; i++) rs_old += r[i] * r[i];
    printf("rs_old: %.3f\n", rs_old);

    dim3 block(16, 32);  // matches PPCG-generated launch config
    dim3 grid(2, 2);

    clock_t start_time = clock();
    int iters = 0;

    for (int k = 0; k < MAX_ITER; k++) {
        iters = k + 1;

        // copy p into the interior of the zero-padded buffer, row by row
        for (int i = 0; i < m; i++) {
            memcpy(p_padded + (i + 1) * pad + 1, p + i * m, m * sizeof(float));
        }
        cudaCheckReturn(cudaMemcpy(dev_p_padded, p_padded, pad * pad * sizeof(float), cudaMemcpyHostToDevice));

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

    // ---- Verify against analytical solution u(x,y) = 1 + x^2 + y^2 ----
    for (int i = 1; i < fulln - 1; i++) {
        for (int j = 1; j < fulln - 1; j++) {
            W[i * fulln + j] = x[j + (i - 1) * m - 1];
        }
    }
    double max_err = 0.0;
    for (int i = 1; i < fulln - 1; i++) {
        xh = i * h;
        for (int j = 1; j < fulln - 1; j++) {
            yh = j * h;
            double exact = 1 + xh * xh + yh * yh;
            double err = fabs(W[i * fulln + j] - exact);
            if (err > max_err) max_err = err;
        }
    }
    printf("Max abs error vs analytical solution: %.6e\n", max_err);
    printf("Time taken for main loop: %.3f ms\n", elapsed_time);

    cudaFree(dev_p_padded); cudaFree(dev_Ap);
    free(W); free(b); free(x); free(r); free(p); free(Ap); free(p_padded);

    return 0;
}
