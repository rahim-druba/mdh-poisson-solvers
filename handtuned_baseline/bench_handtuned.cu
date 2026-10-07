// Hand-written, hand-tuned matrix-free baseline for the Poisson stencil matvec
// Ap = A*p (2D 5-point, diag 4 / 3D 7-point, diag 6, neighbors -1, Dirichlet
// out-of-bounds contributes 0). Reviewer issue #3: an expert-written
// matrix-free ceiling, so MDH's generated kernel can be judged independently of
// the CSR-vs-matrix-free (data representation) question.
//
// Same representation as MDH here: unpadded flat vector, row-major
// (2D: idx = i*C + j ; 3D: idx = (i*M + j)*M + k), fp32, same inputs, same
// CPU oracle, same cudaEvent protocol (>=300 ms warmup + 200 timed launches).
//
// Kernel families (every config below is timed and verified, the best of each
// family is the "hand-tuned" number):
//   naive : one thread per output, plain global loads, branches at boundaries
//   reg   : __ldg loads, each thread walks CY (2D) / CI (3D) consecutive rows /
//           planes along the slow axis keeping the vertical neighbours in
//           registers (sliding window) -- the classic "register blocking"
//   smem  : shared-memory tile + halo for the in-plane neighbours, same
//           slow-axis marching (2.5D blocking in 3D)
//
// Build:  -DDIM=2|3 -DSIDE=<R=C or M>  [-DWITH_MDH + MDH -D macros]
// See build_and_run.sh.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#ifndef DIM
#error "define -DDIM=2 or -DDIM=3"
#endif
#if DIM == 2
#ifndef ROWS
#define ROWS SIDE
#endif
#ifndef COLS
#define COLS SIDE
#endif
#endif
#ifndef SIDE
#define SIDE 0
#endif

#define CUDA_CHECK(e) do { cudaError_t _err = (e); if (_err != cudaSuccess) { \
    fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__); exit(1); } } while (0)

static const int TIMED = 200;
#ifndef REPS
#define REPS 10       // paper protocol: 10 runs, first dropped, 9 averaged
#endif
static const float MIN_WARMUP_MS = 300.0f;
static const double TOL = 1e-3;

#ifdef WITH_MDH
extern __global__ void MDH_KERNEL(float const * const __restrict__ P,
                                  float * const __restrict__ res_g,
                                  float * const __restrict__ int_res);
#endif

// ----------------------------------------------------------------------------
// 2D kernels. S = SIDE x SIDE, idx = i*S + j (j contiguous / x).
// ----------------------------------------------------------------------------
__global__ void st2d_naive(const float* __restrict__ p, float* __restrict__ ap, int R, int C) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= R || j >= C) return;
    float s = 4.0f * p[i * C + j];
    if (i > 0)     s -= p[(i - 1) * C + j];
    if (i < R - 1) s -= p[(i + 1) * C + j];
    if (j > 0)     s -= p[i * C + j - 1];
    if (j < C - 1) s -= p[i * C + j + 1];
    ap[i * C + j] = s;
}

template <int BX, int BY, int CY>
__global__ void st2d_reg(const float* __restrict__ p, float* __restrict__ ap, int R, int C) {
    int j  = blockIdx.x * BX + threadIdx.x;
    int i0 = (blockIdx.y * BY + threadIdx.y) * CY;
    if (j >= C || i0 >= R) return;
    float up  = (i0 > 0) ? __ldg(&p[(i0 - 1) * C + j]) : 0.0f;
    float cen = __ldg(&p[i0 * C + j]);
#pragma unroll
    for (int r = 0; r < CY; ++r) {
        int i = i0 + r;
        if (i >= R) break;
        float dn = (i + 1 < R) ? __ldg(&p[(i + 1) * C + j]) : 0.0f;
        float l  = (j > 0)     ? __ldg(&p[i * C + j - 1]) : 0.0f;
        float rt = (j < C - 1) ? __ldg(&p[i * C + j + 1]) : 0.0f;
        ap[i * C + j] = 4.0f * cen - up - dn - l - rt;
        up = cen; cen = dn;
    }
}

template <int BX, int BY, int CY>
__global__ void st2d_smem(const float* __restrict__ p, float* __restrict__ ap, int R, int C) {
    // each block: BX columns x (BY*CY) rows; each thread walks CY rows of a column
    __shared__ float t[BY + 2][BX + 2];   // only rows used per step: BY (+2 halo rows)
    int tx = threadIdx.x, ty = threadIdx.y;
    int j  = blockIdx.x * BX + tx;
    int ib = blockIdx.y * BY * CY;
    // one "step" = BY rows processed by the block; CY steps per block
    for (int s = 0; s < CY; ++s) {
        int i = ib + s * BY + ty;
        bool in = (i < R && j < C);
        t[ty + 1][tx + 1] = in ? __ldg(&p[i * C + j]) : 0.0f;
        if (ty == 0)      { int ii = ib + s * BY - 1;  t[0][tx + 1]      = (ii >= 0 && ii < R && j < C) ? __ldg(&p[ii * C + j]) : 0.0f; }
        if (ty == BY - 1) { int ii = ib + s * BY + BY; t[BY + 1][tx + 1] = (ii < R && j < C)            ? __ldg(&p[ii * C + j]) : 0.0f; }
        if (tx == 0)      t[ty + 1][0]      = (i < R && j > 0 && j - 1 < C) ? __ldg(&p[i * C + j - 1]) : 0.0f;
        if (tx == BX - 1) t[ty + 1][BX + 1] = (i < R && j + 1 < C)          ? __ldg(&p[i * C + j + 1]) : 0.0f;
        __syncthreads();
        if (in) ap[i * C + j] = 4.0f * t[ty + 1][tx + 1] - t[ty][tx + 1] - t[ty + 2][tx + 1]
                                       - t[ty + 1][tx] - t[ty + 1][tx + 2];
        __syncthreads();
    }
}

// ----------------------------------------------------------------------------
// 3D kernels. M x M x M, idx = (i*M + j)*M + k (k contiguous / x), i slowest.
// ----------------------------------------------------------------------------
__global__ void st3d_naive(const float* __restrict__ p, float* __restrict__ ap, int M) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= M || j >= M || k >= M) return;
    int idx = (i * M + j) * M + k;
    float s = 6.0f * p[idx];
    if (i > 0)     s -= p[idx - M * M];
    if (i < M - 1) s -= p[idx + M * M];
    if (j > 0)     s -= p[idx - M];
    if (j < M - 1) s -= p[idx + M];
    if (k > 0)     s -= p[idx - 1];
    if (k < M - 1) s -= p[idx + 1];
    ap[idx] = s;
}

template <int BX, int BY, int CI>
__global__ void st3d_reg(const float* __restrict__ p, float* __restrict__ ap, int M) {
    int k = blockIdx.x * BX + threadIdx.x;
    int j = blockIdx.y * BY + threadIdx.y;
    int i0 = blockIdx.z * CI;
    if (k >= M || j >= M) return;
    const long MM = (long)M * M;
    const float* col = p + (long)j * M + k;
    float prev = (i0 > 0) ? __ldg(&col[(i0 - 1) * MM]) : 0.0f;
    float cur  = __ldg(&col[i0 * MM]);
#pragma unroll
    for (int s = 0; s < CI; ++s) {
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

template <int BX, int BY, int CI>
__global__ void st3d_smem(const float* __restrict__ p, float* __restrict__ ap, int M) {
    __shared__ float t[BY + 2][BX + 2];
    int tx = threadIdx.x, ty = threadIdx.y;
    int k = blockIdx.x * BX + tx;
    int j = blockIdx.y * BY + ty;
    int i0 = blockIdx.z * CI;
    const long MM = (long)M * M;
    bool in = (k < M && j < M);
    const float* col = p + (long)j * M + k;   // only dereferenced when in
    float prev = (in && i0 > 0) ? __ldg(&col[(i0 - 1) * MM]) : 0.0f;
    float cur  = in ? __ldg(&col[i0 * MM]) : 0.0f;
    for (int s = 0; s < CI; ++s) {
        int i = i0 + s;
        if (i >= M) break;                                   // uniform across the block
        float nxt = (in && i + 1 < M) ? __ldg(&col[(i + 1) * MM]) : 0.0f;
        const float* pl = p + i * MM;
        t[ty + 1][tx + 1] = cur;
        if (ty == 0)      t[0][tx + 1]      = (k < M && j > 0 && j - 1 < M)  ? __ldg(&pl[(long)(j - 1) * M + k]) : 0.0f;
        if (ty == BY - 1) t[BY + 1][tx + 1] = (k < M && j + 1 < M)           ? __ldg(&pl[(long)(j + 1) * M + k]) : 0.0f;
        if (tx == 0)      t[ty + 1][0]      = (j < M && k > 0 && k - 1 < M)  ? __ldg(&pl[(long)j * M + k - 1]) : 0.0f;
        if (tx == BX - 1) t[ty + 1][BX + 1] = (j < M && k + 1 < M)           ? __ldg(&pl[(long)j * M + k + 1]) : 0.0f;
        __syncthreads();
        if (in) ap[i * MM + (long)j * M + k] = 6.0f * cur - prev - nxt
                                              - t[ty][tx + 1] - t[ty + 2][tx + 1]
                                              - t[ty + 1][tx] - t[ty + 1][tx + 2];
        __syncthreads();
        prev = cur; cur = nxt;
    }
}

// ----------------------------------------------------------------------------
// harness
// ----------------------------------------------------------------------------
struct Row { std::string name; double ms; double err; double rsd; };

static float *d_p, *d_ap;
static std::vector<float> h_ref, h_res;
static long N;

template <typename Launch>
static Row time_variant(const std::string& name, Launch launch) {
    CUDA_CHECK(cudaMemset(d_ap, 0, N * sizeof(float)));
    launch();                                         // also catches launch-config errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_res.data(), d_ap, N * sizeof(float), cudaMemcpyDeviceToHost));
    double err = 0.0;
    for (long i = 0; i < N; ++i) err = fmax(err, fabs((double)h_res[i] - h_ref[i]));

    cudaEvent_t ws, we, s, e;
    CUDA_CHECK(cudaEventCreate(&ws)); CUDA_CHECK(cudaEventCreate(&we));
    CUDA_CHECK(cudaEventRecord(ws));
    float warm = 0.0f;
    do { launch(); CUDA_CHECK(cudaEventRecord(we)); CUDA_CHECK(cudaEventSynchronize(we));
         CUDA_CHECK(cudaEventElapsedTime(&warm, ws, we)); } while (warm < MIN_WARMUP_MS);
    CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    std::vector<double> t;
    for (int r = 0; r < REPS; ++r) {
        CUDA_CHECK(cudaEventRecord(s));
        for (int i = 0; i < TIMED; ++i) launch();
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        t.push_back(ms / TIMED);
    }
    CUDA_CHECK(cudaGetLastError());
    cudaEventDestroy(ws); cudaEventDestroy(we); cudaEventDestroy(s); cudaEventDestroy(e);
    double mean = 0, var = 0;
    for (int r = 1; r < REPS; ++r) mean += t[r];
    mean /= (REPS - 1);
    for (int r = 1; r < REPS; ++r) var += (t[r] - mean) * (t[r] - mean);
    double rsd = sqrt(var / (REPS - 2)) / mean * 100.0;
    return {name, mean, err, rsd};
}

static std::vector<Row> rows;
static void report(const Row& r) {
    double gbs = 8.0 * (double)N / (r.ms * 1e-3) / 1e9;   // minimal traffic: read p once + write Ap once
    printf("  %-22s %10.5f ms (+-%.2f%%) %7.1f GB/s  err %.2e  %s\n", r.name.c_str(), r.ms, r.rsd, gbs, r.err,
           r.err < TOL ? "ok" : "WRONG");
    rows.push_back(r);
}

#define RUN2D_REG(BX, BY, CY) { \
    dim3 b(BX, BY), g((C + BX - 1) / BX, (R + BY * CY - 1) / (BY * CY)); \
    report(time_variant("reg   " #BX "x" #BY " cy" #CY, [&] { st2d_reg<BX, BY, CY><<<g, b>>>(d_p, d_ap, R, C); })); }
#define RUN2D_SMEM(BX, BY, CY) { \
    dim3 b(BX, BY), g((C + BX - 1) / BX, (R + BY * CY - 1) / (BY * CY)); \
    report(time_variant("smem  " #BX "x" #BY " cy" #CY, [&] { st2d_smem<BX, BY, CY><<<g, b>>>(d_p, d_ap, R, C); })); }
#define RUN3D_REG(BX, BY, CI) { \
    dim3 b(BX, BY), g((M + BX - 1) / BX, (M + BY - 1) / BY, (M + CI - 1) / CI); \
    report(time_variant("reg   " #BX "x" #BY " ci" #CI, [&] { st3d_reg<BX, BY, CI><<<g, b>>>(d_p, d_ap, M); })); }
#define RUN3D_SMEM(BX, BY, CI) { \
    dim3 b(BX, BY), g((M + BX - 1) / BX, (M + BY - 1) / BY, (M + CI - 1) / CI); \
    report(time_variant("smem  " #BX "x" #BY " ci" #CI, [&] { st3d_smem<BX, BY, CI><<<g, b>>>(d_p, d_ap, M); })); }

int main() {
    const int S = SIDE;
    #if DIM == 2
    N = (long)ROWS * COLS;
#else
    N = (long)S * S * S;
#endif
    #if DIM == 2
    printf("=== hand-tuned matrix-free baseline: Ap = A*p, 2D, %dx%d, N=%ld ===\n", (int)ROWS, (int)COLS, N);
#else
    printf("=== hand-tuned matrix-free baseline: Ap = A*p, 3D, side %d, N=%ld ===\n", S, N);
#endif
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    printf("Methodology: >=%.0f ms warmup + %d timed launches, cudaEvent, averaged; checked vs CPU reference.\n"
           "GB/s = 8*N bytes (minimal traffic) / time.\n\n", MIN_WARMUP_MS, TIMED);

    std::vector<float> h_p(N);
    h_ref.assign(N, 0.0f); h_res.assign(N, 0.0f);
    srand(42);
    for (long i = 0; i < N; ++i) h_p[i] = (float)(rand() % 100) / 10.0f - 5.0f;

#if DIM == 2
    {
        const int R = ROWS, C = COLS;
        for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) {
            float s = 4.0f * h_p[i * C + j];
            if (i > 0)     s -= h_p[(i - 1) * C + j];
            if (i < R - 1) s -= h_p[(i + 1) * C + j];
            if (j > 0)     s -= h_p[i * C + j - 1];
            if (j < C - 1) s -= h_p[i * C + j + 1];
            h_ref[i * C + j] = s;
        }
    }
#else
    {
        const int M = S;
        for (int i = 0; i < M; ++i) for (int j = 0; j < M; ++j) for (int k = 0; k < M; ++k) {
            long idx = ((long)i * M + j) * M + k;
            float s = 6.0f * h_p[idx];
            if (i > 0)     s -= h_p[idx - (long)M * M];
            if (i < M - 1) s -= h_p[idx + (long)M * M];
            if (j > 0)     s -= h_p[idx - M];
            if (j < M - 1) s -= h_p[idx + M];
            if (k > 0)     s -= h_p[idx - 1];
            if (k < M - 1) s -= h_p[idx + 1];
            h_ref[idx] = s;
        }
    }
#endif
    CUDA_CHECK(cudaMalloc(&d_p, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ap, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_p, h_p.data(), N * sizeof(float), cudaMemcpyHostToDevice));

#ifndef MDH_ONLY
    printf("# sweep (every config is timed and verified)\n");
#if DIM == 2
    {
        const int R = ROWS, C = COLS;
        { dim3 b(32, 8), g((C + 31) / 32, (R + 7) / 8);
          report(time_variant("naive 32x8", [&] { st2d_naive<<<g, b>>>(d_p, d_ap, R, C); })); }
        { dim3 b(16, 16), g((C + 15) / 16, (R + 15) / 16);
          report(time_variant("naive 16x16", [&] { st2d_naive<<<g, b>>>(d_p, d_ap, R, C); })); }
        RUN2D_REG(32, 4, 1)  RUN2D_REG(32, 4, 2)  RUN2D_REG(32, 4, 4)  RUN2D_REG(32, 4, 8)
        RUN2D_REG(32, 8, 1)  RUN2D_REG(32, 8, 2)  RUN2D_REG(32, 8, 4)
        RUN2D_REG(64, 4, 1)  RUN2D_REG(64, 4, 2)  RUN2D_REG(64, 4, 4)  RUN2D_REG(64, 4, 8)
        RUN2D_REG(128, 2, 1) RUN2D_REG(128, 2, 2) RUN2D_REG(128, 2, 4) RUN2D_REG(128, 2, 8)
        RUN2D_REG(128, 1, 4) RUN2D_REG(128, 1, 8) RUN2D_REG(256, 1, 4) RUN2D_REG(256, 1, 8)
        RUN2D_SMEM(32, 8, 1) RUN2D_SMEM(32, 8, 2) RUN2D_SMEM(32, 8, 4)
        RUN2D_SMEM(64, 4, 1) RUN2D_SMEM(64, 4, 2) RUN2D_SMEM(64, 4, 4)
        RUN2D_SMEM(32, 16, 1) RUN2D_SMEM(32, 16, 2)
        RUN2D_SMEM(128, 2, 1) RUN2D_SMEM(128, 2, 4)
    }
#else
    {
        const int M = S;
        { dim3 b(32, 4, 2), g((M + 31) / 32, (M + 3) / 4, (M + 1) / 2);
          report(time_variant("naive 32x4x2", [&] { st3d_naive<<<g, b>>>(d_p, d_ap, M); })); }
        { dim3 b(8, 8, 8), g((M + 7) / 8, (M + 7) / 8, (M + 7) / 8);
          report(time_variant("naive 8x8x8", [&] { st3d_naive<<<g, b>>>(d_p, d_ap, M); })); }
        RUN3D_REG(32, 4, 1)  RUN3D_REG(32, 4, 4)  RUN3D_REG(32, 4, 8)  RUN3D_REG(32, 4, 16)
        RUN3D_REG(32, 8, 1)  RUN3D_REG(32, 8, 4)  RUN3D_REG(32, 8, 8)  RUN3D_REG(32, 8, 16)
        RUN3D_REG(64, 4, 4)  RUN3D_REG(64, 4, 8)  RUN3D_REG(64, 4, 16)
        RUN3D_REG(16, 16, 4) RUN3D_REG(16, 16, 8) RUN3D_REG(16, 16, 16)
        RUN3D_REG(32, 16, 4) RUN3D_REG(32, 16, 8)
        RUN3D_SMEM(32, 4, 4)  RUN3D_SMEM(32, 4, 8)  RUN3D_SMEM(32, 4, 16)
        RUN3D_SMEM(32, 8, 4)  RUN3D_SMEM(32, 8, 8)  RUN3D_SMEM(32, 8, 16)
        RUN3D_SMEM(64, 4, 8)  RUN3D_SMEM(16, 16, 8) RUN3D_SMEM(16, 16, 16)
        RUN3D_SMEM(32, 16, 8)
    }
#endif

#endif  // MDH_ONLY

#ifdef WITH_MDH
    printf("\n# MDH-generated kernel (existing config, same harness)\n");
#if DIM == 2
    dim3 mb(NUM_WI_L_2, NUM_WI_L_1), mg(NUM_WG_L_2, NUM_WG_L_1);
#else
    dim3 mb(NUM_WI_L_3, NUM_WI_L_2, NUM_WI_L_1), mg(NUM_WG_L_3, NUM_WG_L_2, NUM_WG_L_1);
#endif
    float* d_res_g; CUDA_CHECK(cudaMalloc(&d_res_g, N * sizeof(float)));
    size_t mdh_first = rows.size();
    report(time_variant("MDH (generated)", [&] { MDH_KERNEL<<<mg, mb>>>(d_p, d_res_g, d_ap); }));
    Row mdh = rows[mdh_first];
#endif

#ifndef MDH_ONLY
    // summary: best correct config per family
    printf("\n# summary (best correct config per family)\n");
    const char* fam[] = {"naive", "reg", "smem"};
    double best_all = 1e30; std::string best_name;
    for (const char* f : fam) {
        const Row* b = nullptr;
        for (const Row& r : rows)
            if (r.name.compare(0, strlen(f), f) == 0 && r.err < TOL && (!b || r.ms < b->ms)) b = &r;
        if (b) {
            printf("  best %-6s %-22s %10.5f ms  %7.1f GB/s\n", f, b->name.c_str(), b->ms,
                   8.0 * (double)N / (b->ms * 1e-3) / 1e9);
            if (strcmp(f, "naive") != 0 && b->ms < best_all) { best_all = b->ms; best_name = b->name; }
        }
    }
    printf("  HAND-TUNED CEILING: %s  %.5f ms\n", best_name.c_str(), best_all);
#endif
#ifdef WITH_MDH
#ifndef MDH_ONLY
    printf("  MDH: %.5f ms (correct: %s)  ->  MDH / hand-tuned = %.2fx\n", mdh.ms,
           mdh.err < TOL ? "yes" : "NO", mdh.ms / best_all);
#else
    printf("  MDH: %.5f ms (correct: %s)\n", mdh.ms, mdh.err < TOL ? "yes" : "NO");
#endif
#endif
    return 0;
}
