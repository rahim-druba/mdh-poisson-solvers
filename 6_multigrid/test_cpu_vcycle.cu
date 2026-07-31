// Pure-CPU V-cycle correctness check, before any GPU code is written.
// Multigrid index/sign/scale bugs are cheap to make and expensive to debug
// on GPU -- verify the scheme actually converges here first.
//
// Grid family m = 2^k - 1 (31, 15, 7, 3, 1), required by the "coarse I <->
// fine 2I+1" alignment in cpu_reference_multigrid_2d.h -- see that file's
// header comment for why. Coarsest level (m=1, a single point) is solved
// exactly in closed form: (1/h^2)*4*u = b -> u = h^2*b/4, no iteration
// needed, no approximation.
//
// Build:
//   nvcc -O3 test_cpu_vcycle.cu -o test_cpu_vcycle
// Run:
//   ./test_cpu_vcycle

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "cpu_reference_multigrid_2d.h"

#define M 31          // finest interior grid side length (2^5 - 1)
#define NU1 2
#define NU2 2
#define OMEGA 0.8f
#define TOL 1e-6
#define MAX_VCYCLES 200

static inline float u_exact(float x, float y) {
    return 1.0f + x * x + y * y;
}

int main() {
    // Levels: m, (m-1)/2, ((m-1)/2-1)/2, ... down to 1.
    int L = 0;
    { int m = M; while (m > 1) { m = (m - 1) / 2; L++; } L++; }
    printf("M=%d, levels=%d\n", M, L);

    int* sizes = (int*)malloc(L * sizeof(int));
    float* h2 = (float*)malloc(L * sizeof(float));
    float** u = (float**)malloc(L * sizeof(float*));
    float** b = (float**)malloc(L * sizeof(float*));
    float** r = (float**)malloc(L * sizeof(float*));

    int fulln = M + 2;
    float h0 = 1.0f / (fulln - 1);

    int m = M;
    for (int l = 0; l < L; l++) {
        sizes[l] = m;
        float h_l = h0 * (float)(1 << l);
        h2[l] = h_l * h_l;
        u[l] = (float*)calloc(m * m, sizeof(float));
        b[l] = (float*)calloc(m * m, sizeof(float));
        r[l] = (float*)calloc(m * m, sizeof(float));
        m = (m - 1) / 2;
    }

    for (int i = 0; i < M; i++) {
        float xh = (i + 1) * h0;
        for (int j = 0; j < M; j++) {
            float yh = (j + 1) * h0;
            float val = -4.0f;
            if (i == 0)     val += u_exact(0.0f, yh) / h2[0];
            if (i == M - 1) val += u_exact(1.0f, yh) / h2[0];
            if (j == 0)     val += u_exact(xh, 0.0f) / h2[0];
            if (j == M - 1) val += u_exact(xh, 1.0f) / h2[0];
            b[0][i * M + j] = val;
        }
    }

    int cycle;
    for (cycle = 0; cycle < MAX_VCYCLES; cycle++) {
        for (int l = 0; l < L - 1; l++) {
            for (int s = 0; s < NU1; s++) cpu_jacobi_smooth(sizes[l], h2[l], b[l], u[l], OMEGA);
            cpu_residual(sizes[l], h2[l], b[l], u[l], r[l]);
            cpu_restrict(sizes[l], r[l], b[l + 1]);
            memset(u[l + 1], 0, sizes[l + 1] * sizes[l + 1] * sizeof(float));
        }
        // coarsest level (m=1): exact closed-form solve
        u[L - 1][0] = h2[L - 1] * b[L - 1][0] / 4.0f;
        for (int l = L - 2; l >= 0; l--) {
            cpu_prolong_add(sizes[l + 1], u[l + 1], u[l]);
            for (int s = 0; s < NU2; s++) cpu_jacobi_smooth(sizes[l], h2[l], b[l], u[l], OMEGA);
        }

        cpu_residual(M, h2[0], b[0], u[0], r[0]);
        double rnorm = 0.0;
        for (int i = 0; i < M * M; i++) rnorm += (double)r[0][i] * r[0][i];
        rnorm = sqrt(rnorm);

        if (cycle < 15) printf("cycle %3d: rnorm=%.6e\n", cycle, rnorm);
        if (rnorm < TOL) { printf("Converged in %d V-cycles, residual norm=%.6e\n", cycle + 1, rnorm); break; }
    }
    if (cycle == MAX_VCYCLES) printf("Did NOT converge in %d V-cycles\n", MAX_VCYCLES);

    double max_err = 0.0;
    for (int i = 0; i < M; i++) {
        float xh = (i + 1) * h0;
        for (int j = 0; j < M; j++) {
            float yh = (j + 1) * h0;
            double exact = u_exact(xh, yh);
            double err = fabs((double)u[0][i * M + j] - exact);
            if (err > max_err) max_err = err;
        }
    }
    printf("Max abs error vs analytical solution: %.6e\n", max_err);
    printf(max_err < 1e-4 ? "PASS\n" : "FAIL\n");

    return max_err < 1e-4 ? 0 : 1;
}
