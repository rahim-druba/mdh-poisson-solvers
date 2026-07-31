// Build:
//   nvcc -O3 kernel_mdh_heat.cu ../2_mdh_sparse/cg_matvec_1.cu -o kernel_mdh_heat \
//     -DTYPE_T=float -DTYPE_TS=float \
//     -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
//     -DG_CB_RES_DEST_LEVEL=2 \
//     -DG_CB_SIZE_L_1=64 -DG_CB_SIZE_L_2=64 \
//     -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=16 -DL_CB_SIZE_L_2=16 \
//     -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1  -DP_CB_SIZE_L_2=1  \
//     -DNUM_WG_L_1=4  -DNUM_WG_L_2=4 \
//     -DNUM_WI_L_1=16 -DNUM_WI_L_2=16 \
//     -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0
// Run:
//   ./kernel_mdh_heat
//
// Real-world application demo: same heat-conduction scenario as
// kernel_sparse_heat.cu, solved with the MDH matrix-free kernel instead,
// so the MDH-generated solver's result on a real problem can be checked
// against an independently-implemented solver. Reuses the exact same compiled
// cg_matvec_1.cu kernel and -D flags already used for N=4096 everywhere
// else in this project -- no kernel changes, only a different RHS.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define FULLN 66
#define TOL 1e-6
#define MAX_ITER 5000
#define SOURCE_AMPLITUDE 100.0f
#define SOURCE_SIGMA 0.1f

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

static inline float heat_source(float x, float y) {
    float dx = x - 0.5f, dy = y - 0.5f;
    return SOURCE_AMPLITUDE * expf(-(dx * dx + dy * dy) / (2.0f * SOURCE_SIGMA * SOURCE_SIGMA));
}

extern __global__ void cg_matvec_1(
    float const * const __restrict__ P,
    float       * const __restrict__ res_g,
    float       * const __restrict__ int_res
);

int main() {
    int fulln = FULLN;
    int m = fulln - 2;
    int N = m * m;
    float h = 1.0f / (fulln - 1);

    if (m != 64) {
        printf("This build's cg_matvec_1 kernel is baked for a 64x64 interior grid.\n");
        return -1;
    }

    printf("N=%d -- heat conduction (MDH matrix-free), Gaussian source at plate center\n", N);

    float* b = (float*)malloc(N * sizeof(float));
    float* x = (float*)calloc(N, sizeof(float));
    float* r = (float*)malloc(N * sizeof(float));
    float* p = (float*)malloc(N * sizeof(float));
    float* Ap = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < m; i++) {
        float xh = (i + 1) * h;
        for (int j = 0; j < m; j++) {
            float yh = (j + 1) * h;
            b[i * m + j] = h * h * heat_source(xh, yh);
        }
    }
    printf("b[center]: %.4f (h^2 * source), max source value: %.2f\n",
           b[(m / 2) * m + m / 2], SOURCE_AMPLITUDE);

    float* dev_p; float* dev_Ap; float* dev_res_g;
    cudaCheckReturn(cudaMalloc((void**)&dev_p, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_Ap, N * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void**)&dev_res_g, N * sizeof(float)));

    for (int i = 0; i < N; i++) { r[i] = b[i]; p[i] = r[i]; }

    float rs_old = 0.0f;
    for (int i = 0; i < N; i++) rs_old += r[i] * r[i];

    dim3 block(16, 16, 1);
    dim3 grid(4, 4, 1);

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
    printf("Time taken for main loop: %.3f ms\n", elapsed_time);

    float max_temp = -1e30f; int max_i = -1, max_j = -1;
    float min_temp = 1e30f;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            float v = x[i * m + j];
            if (v > max_temp) { max_temp = v; max_i = i; max_j = j; }
            if (v < min_temp) min_temp = v;
        }
    printf("Peak temperature %.4f at grid (%d,%d) -> physical (%.3f,%.3f) [expect near (0.5,0.5)]\n",
           max_temp, max_i, max_j, (max_i + 1) * h, (max_j + 1) * h);
    printf("Min temperature in interior: %.4f\n", min_temp);

    FILE* f = fopen("solution_mdh.txt", "w");
    fprintf(f, "%d %d\n", m, m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) fprintf(f, "%.6f ", x[i * m + j]);
        fprintf(f, "\n");
    }
    fclose(f);
    printf("Wrote solution_mdh.txt (%dx%d grid)\n", m, m);

    cudaFree(dev_p); cudaFree(dev_Ap); cudaFree(dev_res_g);
    free(b); free(x); free(r); free(p); free(Ap);

    return 0;
}
