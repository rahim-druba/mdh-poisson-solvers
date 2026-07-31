// Verifies each GPU multigrid kernel against the CPU reference, at m=31
// (and the 31<->15 pair for restrict/prolong) -- before any of these are
// trusted inside a full V-cycle solve.
//
// Build:
//   nvcc -O3 test_gpu_multigrid_kernels.cu -o test_gpu_multigrid_kernels
// Run:
//   ./test_gpu_multigrid_kernels

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cpu_reference_multigrid_2d.h"
#include "gpu_multigrid_kernels.cu"

#define cudaCheckReturn(ret) \
  do { \
    cudaError_t e = (ret); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s (at %s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

int check(const char* name, const float* gpu, const float* cpu, int n) {
    int mismatches = 0;
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(gpu[i] - cpu[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-4f) mismatches++;
    }
    printf("%-20s mismatches=%d max_err=%.3e %s\n", name, mismatches, max_err, mismatches == 0 ? "PASS" : "FAIL");
    return mismatches;
}

int main() {
    int m = 31;
    int fails = 0;
    dim3 block(16, 16);
    dim3 grid((m + 15) / 16, (m + 15) / 16);

    srand(0);
    float* u = (float*)malloc(m * m * sizeof(float));
    float* b = (float*)malloc(m * m * sizeof(float));
    for (int i = 0; i < m * m; i++) { u[i] = (float)(rand() % 100) / 10.0f - 5.0f; b[i] = (float)(rand() % 100) / 10.0f - 5.0f; }
    float h2 = 0.001f;

    float *d_u, *d_b, *d_u2, *d_r;
    cudaCheckReturn(cudaMalloc(&d_u, m * m * sizeof(float)));
    cudaCheckReturn(cudaMalloc(&d_b, m * m * sizeof(float)));
    cudaCheckReturn(cudaMalloc(&d_u2, m * m * sizeof(float)));
    cudaCheckReturn(cudaMalloc(&d_r, m * m * sizeof(float)));
    cudaCheckReturn(cudaMemcpy(d_u, u, m * m * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(d_b, b, m * m * sizeof(float), cudaMemcpyHostToDevice));

    // --- Jacobi smooth ---
    {
        float* u_cpu = (float*)malloc(m * m * sizeof(float));
        memcpy(u_cpu, u, m * m * sizeof(float));
        cpu_jacobi_smooth(m, h2, b, u_cpu, 0.8f);

        jacobi_smooth_kernel<<<grid, block>>>(m, h2, d_b, d_u, d_u2, 0.8f);
        cudaCheckReturn(cudaGetLastError()); cudaCheckReturn(cudaDeviceSynchronize());
        float* u_gpu = (float*)malloc(m * m * sizeof(float));
        cudaCheckReturn(cudaMemcpy(u_gpu, d_u2, m * m * sizeof(float), cudaMemcpyDeviceToHost));
        fails += check("jacobi_smooth", u_gpu, u_cpu, m * m);
        free(u_cpu); free(u_gpu);
    }

    // --- residual ---
    {
        float* r_cpu = (float*)malloc(m * m * sizeof(float));
        cpu_residual(m, h2, b, u, r_cpu);

        residual_kernel<<<grid, block>>>(m, h2, d_b, d_u, d_r);
        cudaCheckReturn(cudaGetLastError()); cudaCheckReturn(cudaDeviceSynchronize());
        float* r_gpu = (float*)malloc(m * m * sizeof(float));
        cudaCheckReturn(cudaMemcpy(r_gpu, d_r, m * m * sizeof(float), cudaMemcpyDeviceToHost));
        fails += check("residual", r_gpu, r_cpu, m * m);
        free(r_cpu); free(r_gpu);
    }

    // --- restrict (31 -> 15) ---
    int mc = 15;
    {
        float* rc_cpu = (float*)calloc(mc * mc, sizeof(float));
        cpu_restrict(m, u, rc_cpu); // reuse u as an arbitrary fine-grid field

        float* d_rc; cudaCheckReturn(cudaMalloc(&d_rc, mc * mc * sizeof(float)));
        dim3 grid_c((mc + 15) / 16, (mc + 15) / 16);
        restrict_kernel<<<grid_c, block>>>(m, d_u, d_rc);
        cudaCheckReturn(cudaGetLastError()); cudaCheckReturn(cudaDeviceSynchronize());
        float* rc_gpu = (float*)malloc(mc * mc * sizeof(float));
        cudaCheckReturn(cudaMemcpy(rc_gpu, d_rc, mc * mc * sizeof(float), cudaMemcpyDeviceToHost));
        fails += check("restrict", rc_gpu, rc_cpu, mc * mc);
        free(rc_cpu); free(rc_gpu); cudaFree(d_rc);
    }

    // --- prolong_add (15 -> 31) ---
    {
        float* ec = (float*)malloc(mc * mc * sizeof(float));
        for (int i = 0; i < mc * mc; i++) ec[i] = (float)(rand() % 100) / 10.0f - 5.0f;
        float* ef_cpu = (float*)malloc(m * m * sizeof(float));
        memcpy(ef_cpu, u, m * m * sizeof(float)); // start from existing field, like the real V-cycle does
        cpu_prolong_add(mc, ec, ef_cpu);

        float* d_ec; cudaCheckReturn(cudaMalloc(&d_ec, mc * mc * sizeof(float)));
        cudaCheckReturn(cudaMemcpy(d_ec, ec, mc * mc * sizeof(float), cudaMemcpyHostToDevice));
        float* d_ef; cudaCheckReturn(cudaMalloc(&d_ef, m * m * sizeof(float)));
        cudaCheckReturn(cudaMemcpy(d_ef, u, m * m * sizeof(float), cudaMemcpyHostToDevice));

        prolong_add_kernel<<<grid, block>>>(mc, d_ec, d_ef);
        cudaCheckReturn(cudaGetLastError()); cudaCheckReturn(cudaDeviceSynchronize());
        float* ef_gpu = (float*)malloc(m * m * sizeof(float));
        cudaCheckReturn(cudaMemcpy(ef_gpu, d_ef, m * m * sizeof(float), cudaMemcpyDeviceToHost));
        fails += check("prolong_add", ef_gpu, ef_cpu, m * m);
        free(ec); free(ef_cpu); free(ef_gpu); cudaFree(d_ec); cudaFree(d_ef);
    }

    printf(fails == 0 ? "\nALL PASS\n" : "\nSOME FAILED\n");
    return fails == 0 ? 0 : 1;
}
