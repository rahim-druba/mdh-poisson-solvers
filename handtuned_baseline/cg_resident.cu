// GPU-resident Conjugate Gradient at LARGE sizes, matrix-free (hand-tuned kernel) vs CSR vs cuSPARSE (reviewer issues #1/#4).
// Same CG loop for all three: the matvec is the only difference; dot products / vector updates are the same fused GPU
// kernels (only the scalars cross the PCIe bus). At these sizes fp32 CG does not reach the absolute 1e-6 tolerance within a
// reasonable number of iterations, so the solver runs a FIXED number of iterations (ITERS) and reports ms per iteration;
// the final residual norm is printed as a correctness check (it must agree between the variants).
// Build: nvcc -O3 -arch=sm_XX cg_resident.cu -DDIM=2 -DROWS=r -DCOLS=c | -DDIM=3 -DSIDE=m  -DBX=.. -DBY=.. -DCY=..  -DMV=0|1|2 -lcusparse
//   MV=0 hand-tuned matrix-free (register-blocked kernel), MV=1 hand-written CSR-scalar, MV=2 cuSPARSE SpMV (CSR)
//   MV=3 MDH-generated matrix-free kernel (link cg_matvec_1.cu / cg_matvec_3d_1.cu, pass the MDH -D macros)
//   MV=4 PPCG-generated kernel (link its kernel.cu, -I its dir, pass -DPBX -DPBY -DPBZ -DPGX -DPGY); PPCG needs a zero-padded
//        input, so p is kept in the padded layout and the vector kernels index into it (no extra copy per iteration)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#ifndef ITERS
#define ITERS 100
#endif
#ifndef REPS
#define REPS 6
#endif
#if MV == 3 && !defined(ST_NUM_WI_L_1) && defined(NUM_WI_L_1)
#define ST_NUM_WI_L_1 NUM_WI_L_1
#define ST_NUM_WG_L_1 NUM_WG_L_1
#define ST_NUM_WI_L_2 NUM_WI_L_2
#define ST_NUM_WG_L_2 NUM_WG_L_2
#ifdef NUM_WI_L_3
#define ST_NUM_WI_L_3 NUM_WI_L_3
#define ST_NUM_WG_L_3 NUM_WG_L_3
#endif
#endif
#if MV == 3
extern __global__ void MDH_KERNEL(float const * const __restrict__ P, float * const __restrict__ res_g, float * const __restrict__ int_res);
#endif
#if MV == 4
extern __global__ void kernel0(float* Ap, float* P);
#endif
#ifndef VEC
#define VEC -1   // -1: fused hand-written vector kernels; 0: unfused hand-written dot/axpy; 1: unfused MDH-generated dot/axpy
#endif
#if VEC == 1
extern __global__ void dot_1(float const * const __restrict__ U, float const * const __restrict__ V, float * const __restrict__ res_g, float * const __restrict__ int_res, float * const __restrict__ S_orig);
extern __global__ void axpy_1(float const * const __restrict__ A, float const * const __restrict__ B, const float s, float * const __restrict__ res_g, float * const __restrict__ int_res);
#endif
#define CK(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) { fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1); } } while (0)
#define CS(e) do { cusparseStatus_t _s = (e); if (_s != CUSPARSE_STATUS_SUCCESS) { fprintf(stderr, "cuSPARSE %d at %d\n", (int)_s, __LINE__); exit(1); } } while (0)
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
#if MV == 4
#if DIM == 2
__device__ __forceinline__ long PI(long i) { long a = i / COLS, b = i % COLS; return (a + 1) * (COLS + 2) + b + 1; }
#else
__device__ __forceinline__ long PI(long i) { long a = i / ((long)SIDE * SIDE), b = (i / SIDE) % SIDE, c = i % SIDE; return ((a + 1) * (SIDE + 2) + (b + 1)) * (SIDE + 2) + c + 1; }
#endif
#else
#define PI(i) (i)
#endif
#define RB 256
#define RBLOCKS 16384
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float sh[RB];
    sh[threadIdx.x] = v; __syncthreads();
    for (int s = RB / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    float r = sh[0]; __syncthreads(); return r;
}
__global__ void dot_kernel(const float* a, const float* b, float* part, int n) {
    float s = 0.f;
    for (int i = blockIdx.x * RB + threadIdx.x; i < n; i += gridDim.x * RB) s += a[PI(i)] * b[i];
    s = block_sum(s); if (threadIdx.x == 0) part[blockIdx.x] = s;
}
// x += alpha p; r -= alpha Ap; partial sums of r.r
// r.r with plain indexing (r is never padded)
__global__ void dot_plain_kernel(const float* a, float* part, int n) {
    float s = 0.f;
    for (int i = blockIdx.x * RB + threadIdx.x; i < n; i += gridDim.x * RB) s += a[i] * a[i];
    s = block_sum(s); if (threadIdx.x == 0) part[blockIdx.x] = s;
}
__global__ void upd_kernel(float* x, float* r, const float* p, const float* Ap, float alpha, float* part, int n) {
    float s = 0.f;
    for (int i = blockIdx.x * RB + threadIdx.x; i < n; i += gridDim.x * RB) {
        x[i] += alpha * p[PI(i)]; float rv = r[i] - alpha * Ap[i]; r[i] = rv; s += rv * rv;
    }
    s = block_sum(s); if (threadIdx.x == 0) part[blockIdx.x] = s;
}
__global__ void p_kernel(float* p, const float* r, float beta, int n) {
    for (int i = blockIdx.x * RB + threadIdx.x; i < n; i += gridDim.x * RB) p[PI(i)] = r[i] + beta * p[PI(i)];
}

#if DIM == 3
static inline float u_exact(float x, float y, float z) { return 1.0f + x * x + y * y + z * z; }
#endif


__global__ void scatter_kernel(const float* src, float* dst, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[PI(i)] = src[i]; }
__global__ void axpy_hw(const float* a, const float* b, float s, float* o, long n) { long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i < n) o[i] = a[i] + s * b[i]; }
__global__ void fill_kernel(float* v, float a, long n) { long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i < n) v[i] = a; }
__global__ void spmv_csr_kernel(int n, const int* __restrict__ rp, const int* __restrict__ ci,
                                const float* __restrict__ v, const float* __restrict__ p, float* __restrict__ ap) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) { float s = 0.f; for (int k = rp[row]; k < rp[row + 1]; ++k) s += v[k] * p[ci[k]]; ap[row] = s; }
}
#if MV == 1 || MV == 2
// CSR of the Poisson operator built on the GPU (same entry order as bench_csr.cu)
#if DIM == 2
__device__ __forceinline__ int row_nnz(long i, long R, long C) { long a = i / C, b = i % C; return 1 + (a > 0) + (b > 0) + (b < C - 1) + (a < R - 1); }
#else
__device__ __forceinline__ int row_nnz(long i, long M, long) { long a = i / (M * M), b = (i / M) % M, c = i % M; return 1 + (a > 0) + (b > 0) + (c > 0) + (c < M - 1) + (b < M - 1) + (a < M - 1); }
#endif
__global__ void count_kernel(int* rp1, long N, long R, long C) { long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i < N) rp1[i] = row_nnz(i, R, C); }
__global__ void fill_csr_kernel(const int* rp, int* ci, float* va, long N, long R, long C) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i >= N) return;
    int k = rp[i];
#if DIM == 2
    long a = i / C, b = i % C;
    if (a > 0)     { ci[k] = (int)(i - C); va[k++] = -1.f; }
    if (b > 0)     { ci[k] = (int)(i - 1); va[k++] = -1.f; }
    ci[k] = (int)i; va[k++] = 4.f;
    if (b < C - 1) { ci[k] = (int)(i + 1); va[k++] = -1.f; }
    if (a < R - 1) { ci[k] = (int)(i + C); va[k++] = -1.f; }
#else
    long M = R; long a = i / (M * M), b = (i / M) % M, c = i % M;
    if (a > 0)     { ci[k] = (int)(i - M * M); va[k++] = -1.f; }
    if (b > 0)     { ci[k] = (int)(i - M);     va[k++] = -1.f; }
    if (c > 0)     { ci[k] = (int)(i - 1);     va[k++] = -1.f; }
    ci[k] = (int)i; va[k++] = 6.f;
    if (c < M - 1) { ci[k] = (int)(i + 1);     va[k++] = -1.f; }
    if (b < M - 1) { ci[k] = (int)(i + M);     va[k++] = -1.f; }
    if (a < M - 1) { ci[k] = (int)(i + M * M); va[k++] = -1.f; }
#endif
}
#endif

int main() {
#if DIM == 2
    const long R = ROWS, C = COLS, N = R * C;
    dim3 block(BX, BY), grid((C + BX - 1) / BX, (R + BY * CY - 1) / (BY * CY));
    #define MATVEC_MF(dp, dap) st_reg<BX, BY, CY><<<grid, block>>>(dp, dap, (int)R, (int)C)
    printf("=== GPU-resident CG, 2D %ldx%ld N=%ld, MV=%d, %d iterations ===\n", R, C, N, MV, ITERS);
#else
    const long R = SIDE, C = SIDE, N = R * R * R;
    dim3 block(BX, BY), grid((R + BX - 1) / BX, (R + BY - 1) / BY, (R + CY - 1) / CY);
    #define MATVEC_MF(dp, dap) st_reg<BX, BY, CY><<<grid, block>>>(dp, dap, (int)R)
    printf("=== GPU-resident CG, 3D side %ld N=%ld, MV=%d, %d iterations ===\n", R, N, MV, ITERS);
#endif
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); printf("device: %s\n", pr.name);
    const int n = (int)N;
#ifdef DOT_WG
    const int nbl = DOT_WG;   // same number of partial sums in every variant (also for the hand-written dot)
#else
    const int nbl = (n + RB - 1) / RB < RBLOCKS ? (n + RB - 1) / RB : RBLOCKS;
#endif
    float *dp, *dAp, *dx, *dr, *part, *db;
#if MV == 4
#if DIM == 2
    const long PN = (R + 2) * (C + 2);
#else
    const long PN = (R + 2) * (R + 2) * (R + 2);
#endif
#else
    const long PN = N;
#endif
    CK(cudaMalloc(&dp, PN * 4)); CK(cudaMemset(dp, 0, PN * 4)); CK(cudaMalloc(&dAp, N * 4)); CK(cudaMalloc(&dx, N * 4)); CK(cudaMalloc(&dr, N * 4)); CK(cudaMalloc(&db, N * 4));
    CK(cudaMalloc(&part, RBLOCKS * 4));
    fill_kernel<<<(n + 255) / 256, 256>>>(db, 1.0f, n);   // right-hand side b = 1
#if MV == 1 || MV == 2
    int *rp, *ci; float* va; long nnz;
    CK(cudaMalloc(&rp, (N + 1) * 4)); CK(cudaMemset(rp, 0, 4));
    count_kernel<<<(n + 255) / 256, 256>>>(rp + 1, N, R, C);
    thrust::inclusive_scan(thrust::device_ptr<int>(rp + 1), thrust::device_ptr<int>(rp + 1 + N), thrust::device_ptr<int>(rp + 1));
    int nz32; CK(cudaMemcpy(&nz32, rp + N, 4, cudaMemcpyDeviceToHost)); nnz = nz32;
    CK(cudaMalloc(&ci, nnz * 4)); CK(cudaMalloc(&va, nnz * 4));
    fill_csr_kernel<<<(n + 255) / 256, 256>>>(rp, ci, va, N, R, C); CK(cudaGetLastError());
#if MV == 2
    cusparseHandle_t h; CS(cusparseCreate(&h));
    cusparseSpMatDescr_t A; CS(cusparseCreateCsr(&A, N, N, nnz, rp, ci, va, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnVecDescr_t vp, vap; CS(cusparseCreateDnVec(&vp, N, dp, CUDA_R_32F)); CS(cusparseCreateDnVec(&vap, N, dAp, CUDA_R_32F));
    const float al = 1.f, be = 0.f; size_t bs = 0;
    CS(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &al, A, vp, &be, vap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bs));
    void* sbuf; CK(cudaMalloc(&sbuf, bs ? bs : 1));
    #define MATVEC(dp_, dap_) CS(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &al, A, vp, &be, vap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, sbuf))
#else
    #define MATVEC(dp_, dap_) spmv_csr_kernel<<<(n + 255) / 256, 256>>>(n, rp, ci, va, dp_, dap_)
#endif
    printf("CSR: nnz=%ld\n", nnz);
#elif MV == 3
    float* dres; CK(cudaMalloc(&dres, N * 4));
#if DIM == 2
    dim3 mb(ST_NUM_WI_L_2, ST_NUM_WI_L_1), mg(ST_NUM_WG_L_2, ST_NUM_WG_L_1);
#else
    dim3 mb(ST_NUM_WI_L_3, ST_NUM_WI_L_2, ST_NUM_WI_L_1), mg(ST_NUM_WG_L_3, ST_NUM_WG_L_2, ST_NUM_WG_L_1);
#endif
    #define MATVEC(dp_, dap_) MDH_KERNEL<<<mg, mb>>>(dp_, dres, dap_)
#elif MV == 4
    dim3 pb(PBX, PBY, PBZ), pg(PGX, PGY);
    #define MATVEC(dp_, dap_) kernel0<<<pg, pb>>>(dap_, dp_)
#else
    #define MATVEC(dp_, dap_) MATVEC_MF(dp_, dap_)
#endif
    std::vector<float> hp(RBLOCKS);
    auto finish = [&](int nb) { CK(cudaMemcpy(hp.data(), part, nb * 4, cudaMemcpyDeviceToHost)); double s = 0; for (int i = 0; i < nb; i++) s += hp[i]; return (float)s; };
#if VEC == 0
    #define DOT(a_, b_) ([&] { dot_kernel<<<nbl, RB>>>(a_, b_, part, n); return finish(nbl); }())
    #define AXPY(a_, b_, s_, o_) axpy_hw<<<(n + 255) / 256, 256>>>(a_, b_, s_, o_, n)
#elif VEC == 1
    float *dresg, *dpartm, *dSo; CK(cudaMalloc(&dresg, 1 << 20)); CK(cudaMalloc(&dpartm, DOT_WG * 4)); CK(cudaMalloc(&dSo, 1 << 12));
    auto finish_m = [&]() { std::vector<float> h(DOT_WG); CK(cudaMemcpy(h.data(), dpartm, DOT_WG * 4, cudaMemcpyDeviceToHost)); double s = 0; for (int i = 0; i < DOT_WG; i++) s += h[i]; return (float)s; };
    #define DOT(a_, b_) ([&] { dot_1<<<dim3(DOT_WG, 1), dim3(DOT_WI, 1)>>>(a_, b_, dresg, dpartm, dSo); return finish_m(); }())
    #define AXPY(a_, b_, s_, o_) axpy_1<<<AX_WG, AX_WI>>>(a_, b_, s_, dresg, o_)
#endif
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<double> t; float last_rs = 0;
    for (int rep = 0; rep < REPS; ++rep) {
        CK(cudaMemcpy(dr, db, N * 4, cudaMemcpyDeviceToDevice)); CK(cudaMemset(dx, 0, N * 4));
#if MV == 4
        CK(cudaMemset(dp, 0, PN * 4)); scatter_kernel<<<(n + 255) / 256, 256>>>(db, dp, n);
#else
        CK(cudaMemcpy(dp, db, N * 4, cudaMemcpyDeviceToDevice));
#endif
        CK(cudaDeviceSynchronize());
        dot_plain_kernel<<<nbl, RB>>>(dr, part, n); float rs_old = finish(nbl);
        CK(cudaEventRecord(e0));
#if VEC >= 0
        for (int k = 0; k < ITERS; k++) {
            MATVEC(dp, dAp); CK(cudaGetLastError());
            float pAp = DOT(dp, dAp);
            float alpha = rs_old / pAp;
            AXPY(dx, dp, alpha, dx);      // x += alpha p
            AXPY(dr, dAp, -alpha, dr);    // r -= alpha Ap
            float rs_new = DOT(dr, dr);
            float beta = rs_new / rs_old;
            AXPY(dr, dp, beta, dp);       // p = r + beta p
            rs_old = rs_new;
        }
#else
        for (int k = 0; k < ITERS; k++) {
            MATVEC(dp, dAp); CK(cudaGetLastError());
            dot_kernel<<<nbl, RB>>>(dp, dAp, part, n);
            float pAp = finish(nbl);
            float alpha = rs_old / pAp;
            upd_kernel<<<nbl, RB>>>(dx, dr, dp, dAp, alpha, part, n);
            float rs_new = finish(nbl);
            float beta = rs_new / rs_old;
            p_kernel<<<nbl, RB>>>(dp, dr, beta, n);
            rs_old = rs_new;
        }
#endif
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms / ITERS); last_rs = rs_old;
    }
    double m = 0, v = 0; for (int r = 1; r < REPS; ++r) m += t[r]; m /= (REPS - 1);
    for (int r = 1; r < REPS; ++r) v += (t[r] - m) * (t[r] - m);
    printf("cg-resident MV=%d VEC=%d  %10.5f ms/iter (+-%.2f%%)   residual norm after %d iterations: %.6e\n", MV, VEC, m, sqrt(v / (REPS - 2)) / m * 100, ITERS, sqrt((double)last_rs));
    return 0;
}
