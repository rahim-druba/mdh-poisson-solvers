// Build (M = finest interior grid side length, must be 2^k - 1: 31/63/127/255):
//   nvcc -O3 -DM=31 kernel_multigrid_2d.cu -o kernel_multigrid_2d_M31
// Run:
//   ./kernel_multigrid_2d_M31
//
// Geometric multigrid (V-cycle) solver for the same 2D Poisson problem
// every other solver in this project solves -- domain [0,1]^2, Dirichlet
// BC, analytical solution u = 1 + x^2 + y^2, 5-point stencil. Built to
// give CG a real point of comparison against a different algorithm, not
// another CG variant.
//
// Unlike the CG solvers here, multigrid needs a properly h-scaled operator
// at every level (see cpu_reference_multigrid_2d.h's header comment), so
// this does NOT reuse kernel_sparse.cu's "unscaled stencil, h^2 in the
// RHS only" convention -- it has its own, verified independently
// (test_cpu_vcycle.cu, test_gpu_multigrid_kernels.cu) before being wired
// up here. Grid family is m = 2^k - 1 (not the CG tables' plain powers of
// 2) because that's what keeps every level's boundary distance consistent
// with its own grid spacing -- see that same header comment for the actual
// bug this fixed.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include "gpu_multigrid_kernels.cu"

#ifndef M
#error "M must be defined (-DM=.. finest interior grid side length, 2^k - 1)"
#endif
#define NU1 2
#define NU2 2
#define OMEGA 0.8f
// Relative residual, not absolute -- see header comment. 1e-6 hits
// float32's precision floor before converging at larger M (confirmed
// empirically: capped out at 60+ cycles at M=255 without reaching it,
// despite the solution error already being at its float32 floor). 1e-5
// converges cleanly at every tested size and still gives ~1e-4 solution
// accuracy.
#define REL_TOL 1e-5
#define MAX_VCYCLES 20

#define cudaCheckReturn(ret) \
  do { \
    cudaError_t e = (ret); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s (at %s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

static inline float u_exact(float x, float y) {
    return 1.0f + x * x + y * y;
}

int main() {
    int L = 0;
    { int m = M; while (m > 1) { m = (m - 1) / 2; L++; } L++; }
    printf("M=%d, levels=%d\n", M, L);

    int* sizes = (int*)malloc(L * sizeof(int));
    float* h2 = (float*)malloc(L * sizeof(float));
    int fulln = M + 2;
    float h0 = 1.0f / (fulln - 1);

    int m = M;
    for (int l = 0; l < L; l++) {
        sizes[l] = m;
        float h_l = h0 * (float)(1 << l);
        h2[l] = h_l * h_l;
        m = (m - 1) / 2;
    }

    // ---- device buffers per level: u (ping-pong), b, r ----
    float **d_u = (float**)malloc(L * sizeof(float*));
    float **d_u2 = (float**)malloc(L * sizeof(float*));
    float **d_b = (float**)malloc(L * sizeof(float*));
    float **d_r = (float**)malloc(L * sizeof(float*));
    for (int l = 0; l < L; l++) {
        int n = sizes[l] * sizes[l];
        cudaCheckReturn(cudaMalloc(&d_u[l], n * sizeof(float)));
        cudaCheckReturn(cudaMalloc(&d_u2[l], n * sizeof(float)));
        cudaCheckReturn(cudaMalloc(&d_b[l], n * sizeof(float)));
        cudaCheckReturn(cudaMalloc(&d_r[l], n * sizeof(float)));
        cudaCheckReturn(cudaMemset(d_u[l], 0, n * sizeof(float)));
    }

    // ---- finest-level RHS (host, then copy) ----
    float* b0 = (float*)malloc(M * M * sizeof(float));
    for (int i = 0; i < M; i++) {
        float xh = (i + 1) * h0;
        for (int j = 0; j < M; j++) {
            float yh = (j + 1) * h0;
            float val = -4.0f;
            if (i == 0)     val += u_exact(0.0f, yh) / h2[0];
            if (i == M - 1) val += u_exact(1.0f, yh) / h2[0];
            if (j == 0)     val += u_exact(xh, 0.0f) / h2[0];
            if (j == M - 1) val += u_exact(xh, 1.0f) / h2[0];
            b0[i * M + j] = val;
        }
    }
    printf("b[0]: %.3f\n", b0[0]);
    cudaCheckReturn(cudaMemcpy(d_b[0], b0, M * M * sizeof(float), cudaMemcpyHostToDevice));

    float* h_r0 = (float*)malloc(M * M * sizeof(float));
    dim3 block(16, 16);

    double rnorm0 = 0.0;
    for (int i = 0; i < M * M; i++) rnorm0 += (double)b0[i] * b0[i];
    rnorm0 = sqrt(rnorm0); // u starts at 0, so initial residual == b

    clock_t start_time = clock();
    int cycle;

    for (cycle = 0; cycle < MAX_VCYCLES; cycle++) {
        // ---- down-sweep ----
        for (int l = 0; l < L - 1; l++) {
            dim3 grid_l((sizes[l] + 15) / 16, (sizes[l] + 15) / 16);
            for (int s = 0; s < NU1; s++) {
                jacobi_smooth_kernel<<<grid_l, block>>>(sizes[l], h2[l], d_b[l], d_u[l], d_u2[l], OMEGA);
                float* tmp = d_u[l]; d_u[l] = d_u2[l]; d_u2[l] = tmp;
            }
            residual_kernel<<<grid_l, block>>>(sizes[l], h2[l], d_b[l], d_u[l], d_r[l]);
            dim3 grid_c((sizes[l + 1] + 15) / 16, (sizes[l + 1] + 15) / 16);
            restrict_kernel<<<grid_c, block>>>(sizes[l], d_r[l], d_b[l + 1]);
            cudaCheckReturn(cudaMemset(d_u[l + 1], 0, sizes[l + 1] * sizes[l + 1] * sizeof(float)));
        }
        // ---- coarsest level (m=1): exact closed-form solve ----
        {
            float b_coarse, u_coarse;
            cudaCheckReturn(cudaMemcpy(&b_coarse, d_b[L - 1], sizeof(float), cudaMemcpyDeviceToHost));
            u_coarse = h2[L - 1] * b_coarse / 4.0f;
            cudaCheckReturn(cudaMemcpy(d_u[L - 1], &u_coarse, sizeof(float), cudaMemcpyHostToDevice));
        }
        // ---- up-sweep ----
        for (int l = L - 2; l >= 0; l--) {
            dim3 grid_l((sizes[l] + 15) / 16, (sizes[l] + 15) / 16);
            prolong_add_kernel<<<grid_l, block>>>(sizes[l + 1], d_u[l + 1], d_u[l]);
            for (int s = 0; s < NU2; s++) {
                jacobi_smooth_kernel<<<grid_l, block>>>(sizes[l], h2[l], d_b[l], d_u[l], d_u2[l], OMEGA);
                float* tmp = d_u[l]; d_u[l] = d_u2[l]; d_u2[l] = tmp;
            }
        }

        dim3 grid0((M + 15) / 16, (M + 15) / 16);
        residual_kernel<<<grid0, block>>>(M, h2[0], d_b[0], d_u[0], d_r[0]);
        cudaCheckReturn(cudaGetLastError());
        cudaCheckReturn(cudaMemcpy(h_r0, d_r[0], M * M * sizeof(float), cudaMemcpyDeviceToHost));

        double rnorm = 0.0;
        for (int i = 0; i < M * M; i++) rnorm += (double)h_r0[i] * h_r0[i];
        rnorm = sqrt(rnorm);

        if (rnorm / rnorm0 < REL_TOL) break;
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) * 1000.0 / CLOCKS_PER_SEC;

    printf("Converged in %d V-cycles\n", cycle + 1);

    float* u_final = (float*)malloc(M * M * sizeof(float));
    cudaCheckReturn(cudaMemcpy(u_final, d_u[0], M * M * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int i = 0; i < M; i++) {
        float xh = (i + 1) * h0;
        for (int j = 0; j < M; j++) {
            float yh = (j + 1) * h0;
            double exact = u_exact(xh, yh);
            double err = fabs((double)u_final[i * M + j] - exact);
            if (err > max_err) max_err = err;
        }
    }
    printf("Max abs error vs analytical solution: %.6e\n", max_err);
    printf("Time taken for main loop: %.3f ms\n", elapsed_time);

    for (int l = 0; l < L; l++) { cudaFree(d_u[l]); cudaFree(d_u2[l]); cudaFree(d_b[l]); cudaFree(d_r[l]); }
    free(sizes); free(h2); free(d_u); free(d_u2); free(d_b); free(d_r);
    free(b0); free(h_r0); free(u_final);

    return 0;
}
