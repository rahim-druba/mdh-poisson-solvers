// Hand-written matrix-free Conjugate Gradient solver for the Poisson problem (reviewer issue #3, full-solve tables).
// Same problem setup, initial guess, tolerance (absolute residual norm 1e-6), MAX_ITER, fp32 arithmetic, clock()-based
// timing of the main loop and analytical-error check as the paper's solvers (tables/full_cg/kernel_mdh_cg.cu,
// 5_3d_extension/kernel_mdh_3d.cu); only the matvec kernel is the expert-tuned register-blocked kernel from bench_handtuned.cu.
//
// MODE=0 "host":   identical solver structure to the paper's (copy p to GPU, matvec on GPU, copy Ap back, dot products and
//                  AXPY in host loops). Differs from the MDH solver ONLY in the matvec kernel -> isolates the generator effect.
// MODE=1 "device": fully GPU-resident CG (matvec, fused dot/AXPY kernels; only scalars cross the PCIe bus). The best a
//                  hand-written matrix-free solver can do; shows how much of the paper's solve time is host overhead.
//
// Build: nvcc -O3 -arch=sm_XX cg_handtuned.cu -DDIM=2 -DROWS=r -DCOLS=c -DBX=32 -DBY=4 -DCY=1 -DMODE=0
//        nvcc -O3 -arch=sm_XX cg_handtuned.cu -DDIM=3 -DSIDE=m        -DBX=32 -DBY=8 -DCY=1 -DMODE=1   (CY = CI in 3D)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <cuda_runtime.h>

#define TOL 1e-6
#define MAX_ITER 5000
#define CK(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) { fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1); } } while (0)

#if DIM == 2
template <int bx, int by, int cy>
__global__ void st_reg(const float* __restrict__ p, float* __restrict__ ap, int R, int C) {
    int j  = blockIdx.x * bx + threadIdx.x;
    int i0 = (blockIdx.y * by + threadIdx.y) * cy;
    if (j >= C || i0 >= R) return;
    float up  = (i0 > 0) ? __ldg(&p[(i0 - 1) * C + j]) : 0.0f;
    float cen = __ldg(&p[i0 * C + j]);
#pragma unroll
    for (int r = 0; r < cy; ++r) {
        int i = i0 + r;
        if (i >= R) break;
        float dn = (i + 1 < R) ? __ldg(&p[(i + 1) * C + j]) : 0.0f;
        float l  = (j > 0)     ? __ldg(&p[i * C + j - 1]) : 0.0f;
        float rt = (j < C - 1) ? __ldg(&p[i * C + j + 1]) : 0.0f;
        ap[i * C + j] = 4.0f * cen - up - dn - l - rt;
        up = cen; cen = dn;
    }
}
#else
template <int bx, int by, int ci>
__global__ void st_reg(const float* __restrict__ p, float* __restrict__ ap, int M) {
    int k = blockIdx.x * bx + threadIdx.x;
    int j = blockIdx.y * by + threadIdx.y;
    int i0 = blockIdx.z * ci;
    if (k >= M || j >= M) return;
    const long MM = (long)M * M;
    const float* col = p + (long)j * M + k;
    float prev = (i0 > 0) ? __ldg(&col[(i0 - 1) * MM]) : 0.0f;
    float cur  = __ldg(&col[i0 * MM]);
#pragma unroll
    for (int s = 0; s < ci; ++s) {
        int i = i0 + s;
        if (i >= M) break;
        float nxt = (i + 1 < M) ? __ldg(&col[(i + 1) * MM]) : 0.0f;
        float jm = (j > 0)     ? __ldg(&col[(i * MM) - M]) : 0.0f;
        float jp = (j < M - 1) ? __ldg(&col[(i * MM) + M]) : 0.0f;
        float km = (k > 0)     ? __ldg(&col[(i * MM) - 1]) : 0.0f;
        float kp = (k < M - 1) ? __ldg(&col[(i * MM) + 1]) : 0.0f;
        ap[i * MM + (long)j * M + k] = 6.0f * cur - prev - nxt - jm - jp - km - kp;
        prev = cur; cur = nxt;
    }
}
#endif

// ---- device-mode helper kernels (fp32 block reductions, partial sums finished on the host) ----
#define RB 256
#define RBLOCKS 1024
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float sh[RB];
    sh[threadIdx.x] = v; __syncthreads();
    for (int s = RB / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    float r = sh[0]; __syncthreads(); return r;
}
__global__ void dot_kernel(const float* a, const float* b, float* part, int n) {
    float s = 0.f;
    for (int i = blockIdx.x * RB + threadIdx.x; i < n; i += gridDim.x * RB) s += a[i] * b[i];
    s = block_sum(s); if (threadIdx.x == 0) part[blockIdx.x] = s;
}
// x += alpha p; r -= alpha Ap; partial sums of r.r
__global__ void upd_kernel(float* x, float* r, const float* p, const float* Ap, float alpha, float* part, int n) {
    float s = 0.f;
    for (int i = blockIdx.x * RB + threadIdx.x; i < n; i += gridDim.x * RB) {
        x[i] += alpha * p[i]; float rv = r[i] - alpha * Ap[i]; r[i] = rv; s += rv * rv;
    }
    s = block_sum(s); if (threadIdx.x == 0) part[blockIdx.x] = s;
}
__global__ void p_kernel(float* p, const float* r, float beta, int n) {
    for (int i = blockIdx.x * RB + threadIdx.x; i < n; i += gridDim.x * RB) p[i] = r[i] + beta * p[i];
}

#if DIM == 3
static inline float u_exact(float x, float y, float z) { return 1.0f + x * x + y * y + z * z; }
#endif

int main() {
#if DIM == 2
    const int R = ROWS, C = COLS, N = R * C;
    const int fulln_r = R + 2, fulln_c = C + 2;
    float h = 1.0f / (fulln_c - 1);
    printf("N=%d (%dx%d) hand-tuned matrix-free CG, mode %d, reg %dx%d cy%d\n", N, R, C, MODE, BX, BY, CY);
#else
    const int m = SIDE, N = m * m * m;
    float h = 1.0f / (m + 1);
    printf("N=%d (%d^3) hand-tuned matrix-free CG, mode %d, reg %dx%d ci%d\n", N, m, MODE, BX, BY, CY);
#endif
    std::vector<float> b(N), x(N, 0.f), r(N), p(N), Ap(N);
#if DIM == 2
    std::vector<float> W(fulln_r * fulln_c);
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
    for (int i = 0; i < R; i++) for (int j = 0; j < C; j++) {
        b[i * C + j] = h * h * (-4);
        if (i == 0) b[i * C + j] += W[0 * fulln_c + (j + 1)];
        if (i == R - 1) b[i * C + j] += W[(fulln_r - 1) * fulln_c + (j + 1)];
        if (j == 0) b[i * C + j] += W[(i + 1) * fulln_c + 0];
        if (j == C - 1) b[i * C + j] += W[(i + 1) * fulln_c + (fulln_c - 1)];
    }
    dim3 block(BX, BY), grid((C + BX - 1) / BX, (R + BY * CY - 1) / (BY * CY));
    #define MATVEC(dp, dap) st_reg<BX, BY, CY><<<grid, block>>>(dp, dap, R, C)
#else
    for (int i = 0; i < m; i++) { float xh = (i + 1) * h;
        for (int j = 0; j < m; j++) { float yh = (j + 1) * h;
            for (int k = 0; k < m; k++) { float zh = (k + 1) * h;
                int idx = (i * m + j) * m + k;
                b[idx] = h * h * (-6.0f);
                if (i == 0)     b[idx] += u_exact(0.0f, yh, zh);
                if (i == m - 1) b[idx] += u_exact(1.0f, yh, zh);
                if (j == 0)     b[idx] += u_exact(xh, 0.0f, zh);
                if (j == m - 1) b[idx] += u_exact(xh, 1.0f, zh);
                if (k == 0)     b[idx] += u_exact(xh, yh, 0.0f);
                if (k == m - 1) b[idx] += u_exact(xh, yh, 1.0f);
            } } }
    dim3 block(BX, BY), grid((m + BX - 1) / BX, (m + BY - 1) / BY, (m + CY - 1) / CY);
    #define MATVEC(dp, dap) st_reg<BX, BY, CY><<<grid, block>>>(dp, dap, m)
#endif
    float *dp, *dAp;
    CK(cudaMalloc(&dp, N * 4)); CK(cudaMalloc(&dAp, N * 4));
    for (int i = 0; i < N; i++) { r[i] = b[i]; p[i] = r[i]; }
    float rs_old = 0.0f;
    for (int i = 0; i < N; i++) rs_old += r[i] * r[i];
    int iters = 0;
    clock_t t0;

#if MODE == 0
    t0 = clock();
    for (int k = 0; k < MAX_ITER; k++) {
        iters = k + 1;
        CK(cudaMemcpy(dp, p.data(), N * 4, cudaMemcpyHostToDevice));
        MATVEC(dp, dAp); CK(cudaGetLastError());
        CK(cudaMemcpy(Ap.data(), dAp, N * 4, cudaMemcpyDeviceToHost));
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
    clock_t t1 = clock();
#else
    float *dx, *dr, *part; std::vector<float> hp(RBLOCKS);
    CK(cudaMalloc(&dx, N * 4)); CK(cudaMalloc(&dr, N * 4)); CK(cudaMalloc(&part, RBLOCKS * 4));
    int nbl = (N + RB - 1) / RB; if (nbl > RBLOCKS) nbl = RBLOCKS;
    auto finish = [&](int nb) { CK(cudaMemcpy(hp.data(), part, nb * 4, cudaMemcpyDeviceToHost)); double s = 0; for (int i = 0; i < nb; i++) s += hp[i]; return (float)s; };
    t0 = clock();
    CK(cudaMemcpy(dp, p.data(), N * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dr, r.data(), N * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dx, 0, N * 4));
    for (int k = 0; k < MAX_ITER; k++) {
        iters = k + 1;
        MATVEC(dp, dAp); CK(cudaGetLastError());
        dot_kernel<<<nbl, RB>>>(dp, dAp, part, N);
        float pAp = finish(nbl);
        float alpha = rs_old / pAp;
        upd_kernel<<<nbl, RB>>>(dx, dr, dp, dAp, alpha, part, N);
        float rs_new = finish(nbl);
        if (sqrt(rs_new) < TOL) { rs_old = rs_new; break; }
        float beta = rs_new / rs_old;
        p_kernel<<<nbl, RB>>>(dp, dr, beta, N);
        rs_old = rs_new;
    }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(x.data(), dx, N * 4, cudaMemcpyDeviceToHost));
    clock_t t1 = clock();
#endif
    double ms = (double)(t1 - t0) * 1000.0 / CLOCKS_PER_SEC;
    printf("Converged in %d iterations, final residual norm: %.6e\n", iters, sqrt((double)rs_old));
    double max_err = 0.0;
#if DIM == 2
    for (int i = 0; i < R; i++) { float y_ = (i + 1) * h;
        for (int j = 0; j < C; j++) { float x_ = (j + 1) * h;
            double exact = 1 + (double)x_ * x_ + (double)y_ * y_;
            double err = fabs((double)x[i * C + j] - exact); if (err > max_err) max_err = err; } }
#else
    for (int i = 0; i < m; i++) { float xh = (i + 1) * h;
        for (int j = 0; j < m; j++) { float yh = (j + 1) * h;
            for (int k = 0; k < m; k++) { float zh = (k + 1) * h;
                double err = fabs((double)x[(i * m + j) * m + k] - (double)u_exact(xh, yh, zh)); if (err > max_err) max_err = err; } } }
#endif
    printf("Max abs error vs analytical solution: %.6e\n", max_err);
    printf("Time taken for main loop: %.3f ms\n", ms);
    return 0;
}
