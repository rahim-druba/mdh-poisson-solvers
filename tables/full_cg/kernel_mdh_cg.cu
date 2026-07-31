#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Generalized version of ../../2_mdh_sparse/kernel_mdh_sparse.cu for a
// ROWS x COLS interior grid. Launch dims (NUM_WG_L_1/2, NUM_WI_L_1/2) are
// the same compile-time macros the generator itself uses -- see
// ../matvec/kernel_mdh_cg build commands / build_and_run_sweep.sh.
#ifndef ROWS
#error "ROWS must be defined"
#endif
#ifndef COLS
#error "COLS must be defined"
#endif
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

extern __global__ void cg_matvec_1(
    float const * const __restrict__ P,
    float       * const __restrict__ res_g,
    float       * const __restrict__ int_res
);

int main() {
    const int R = ROWS, C = COLS, N = R * C;
    const int fulln_r = R + 2, fulln_c = C + 2;
    float h = 1.0f / (fulln_c - 1);

    printf("N=%d (%dx%d)\n", N, R, C);

    float* W = (float*)malloc(fulln_r * fulln_c * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));
    float* x = (float*)calloc(N, sizeof(float));
    float* r = (float*)malloc(N * sizeof(float));
    float* p = (float*)malloc(N * sizeof(float));
    float* Ap = (float*)malloc(N * sizeof(float));

    for (int col = 0; col < fulln_c; col++) {
        float x_ = col * h;
        W[0 * fulln_c + col] = 1 + x_ * x_;
        W[(fulln_r - 1) * fulln_c + col] = 1 + x_ * x_ + ((fulln_r - 1) * h) * ((fulln_r - 1) * h);
    }
    for (int row = 0; row < fulln_r; row++) {
        float y_ = row * h;
        W[row * fulln_c + 0] = 1 + y_ * y_;
        W[row * fulln_c + (fulln_c - 1)] = 1 + ((fulln_c - 1) * h) * ((fulln_c - 1) * h) + y_ * y_;
    }
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            b[i * C + j] = h * h * (-4);
            if (i == 0) b[i * C + j] += W[0 * fulln_c + (j + 1)];
            if (i == R - 1) b[i * C + j] += W[(fulln_r - 1) * fulln_c + (j + 1)];
            if (j == 0) b[i * C + j] += W[(i + 1) * fulln_c + 0];
            if (j == C - 1) b[i * C + j] += W[(i + 1) * fulln_c + (fulln_c - 1)];
        }
    }

    float* dev_p; float* dev_Ap; float* dev_res_g;
    cudaCheckReturn(cudaMalloc((void**)&dev_p, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_Ap, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_res_g, N * sizeof(float)));

    for (int i = 0; i < N; i++) { r[i] = b[i]; p[i] = r[i]; }
    float rs_old = 0.0f;
    for (int i = 0; i < N; i++) rs_old += r[i] * r[i];

    dim3 block(NUM_WI_L_2, NUM_WI_L_1, 1);
    dim3 grid(NUM_WG_L_2, NUM_WG_L_1, 1);

    clock_t start_time = clock();
    int iters = 0;
    for (int k = 0; k < MAX_ITER; k++) {
        iters = k + 1;
        cudaCheckReturn(cudaMemcpy(dev_p, p, N * sizeof(float), cudaMemcpyHostToDevice));
        cg_matvec_1<<<grid, block>>>(dev_p, dev_res_g, dev_Ap);
        cudaCheckKernel();
        cudaCheckReturn(cudaMemcpy(Ap, dev_Ap, N * sizeof(float), cudaMemcpyDeviceToHost));

        float pAp = 0.0f;
        for (int i = 0; i < N; i++) pAp += p[i] * Ap[i];
        float alpha = rs_old / pAp;
        for (int i = 0; i < N; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
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
    for (int i = 0; i < R; i++) {
        float y_ = (i + 1) * h;
        for (int j = 0; j < C; j++) {
            float x_ = (j + 1) * h;
            double exact = 1 + (double)x_ * x_ + (double)y_ * y_;
            double err = fabs((double)x[i * C + j] - exact);
            if (err > max_err) max_err = err;
        }
    }
    printf("Max abs error vs analytical solution: %.6e\n", max_err);
    printf("Time taken for main loop: %.3f ms\n", elapsed_time);

    cudaFree(dev_p); cudaFree(dev_Ap); cudaFree(dev_res_g);
    free(W); free(b); free(x); free(r); free(p); free(Ap);
    return 0;
}
