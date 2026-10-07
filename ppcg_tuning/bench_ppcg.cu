// Times a PPCG-generated Poisson-stencil matvec kernel0 (from ppcg_gen.sh) with the same input, CPU oracle and
// cudaEvent protocol as bench_handtuned.cu: >=300 ms warm-up, REPS repetitions (first dropped), mean of the rest.
// Number of timed launches per repetition: 200, reduced for slow kernels so a repetition lasts about 100 ms
// (TIMED = clamp(100 ms / launch time, 10, 200)); the value used is printed.
// Build: nvcc -O3 bench_ppcg.cu gen/<d>/kernel.cu -I gen/<d> -DDIM=2 -DROWS=R -DCOLS=C | -DDIM=3 -DSIDE=M
//        -DBX=.. -DBY=.. -DBZ=.. -DGX=.. -DGY=..   [-DREPS=10]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) { fprintf(stderr, "CUDA '%s' at %d\n", cudaGetErrorString(_e), __LINE__); exit(1); } } while (0)
static int REPS = 10;
extern __global__ void kernel0(float* Ap, float* P);
int main() {
    if (const char* e = getenv("REPS")) REPS = atoi(e);
#if DIM == 2
    const long R = ROWS, C = COLS, N = R * C; const long PN = (R + 2) * (C + 2);
#else
    const long M = SIDE, N = M * M * M; const long PN = (M + 2) * (M + 2) * (M + 2);
#endif
    std::vector<float> p(N), ref(N), res(N), pp(PN, 0.f);
    srand(42); for (long i = 0; i < N; ++i) p[i] = (float)(rand() % 100) / 10.0f - 5.0f;
#if DIM == 2
    for (long i = 0; i < R; ++i) for (long j = 0; j < C; ++j) {
        pp[(i + 1) * (C + 2) + j + 1] = p[i * C + j];
        float s = 4.f * p[i * C + j];
        if (i > 0) s -= p[(i - 1) * C + j]; if (i < R - 1) s -= p[(i + 1) * C + j];
        if (j > 0) s -= p[i * C + j - 1]; if (j < C - 1) s -= p[i * C + j + 1];
        ref[i * C + j] = s; }
#else
    for (long i = 0; i < M; ++i) for (long j = 0; j < M; ++j) for (long k = 0; k < M; ++k) {
        long id = (i * M + j) * M + k; pp[((i + 1) * (M + 2) + j + 1) * (M + 2) + k + 1] = p[id];
        float s = 6.f * p[id];
        if (i > 0) s -= p[id - M * M]; if (i < M - 1) s -= p[id + M * M];
        if (j > 0) s -= p[id - M]; if (j < M - 1) s -= p[id + M];
        if (k > 0) s -= p[id - 1]; if (k < M - 1) s -= p[id + 1];
        ref[id] = s; }
#endif
    float *dP, *dAp; CK(cudaMalloc(&dP, PN * 4)); CK(cudaMalloc(&dAp, N * 4));
    CK(cudaMemcpy(dP, pp.data(), PN * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(dAp, 0, N * 4));
    dim3 block(BX, BY, BZ), grid(GX, GY);
    auto launch = [&] { kernel0<<<grid, block>>>(dAp, dP); };
    launch(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(res.data(), dAp, N * 4, cudaMemcpyDeviceToHost));
    double err = 0; for (long i = 0; i < N; ++i) err = fmax(err, fabs((double)res[i] - ref[i]));
    cudaEvent_t ws, we, s, e; CK(cudaEventCreate(&ws)); CK(cudaEventCreate(&we)); CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    CK(cudaEventRecord(ws)); float w = 0; int nw = 0;
    do { launch(); ++nw; CK(cudaEventRecord(we)); CK(cudaEventSynchronize(we)); CK(cudaEventElapsedTime(&w, ws, we)); } while (w < 300.f);
    double t1 = w / nw;
    int TIMED = (int)std::min(200.0, std::max(10.0, std::ceil(100.0 / t1)));
    std::vector<double> t;
    for (int r = 0; r < REPS; ++r) {
        CK(cudaEventRecord(s)); for (int i = 0; i < TIMED; ++i) launch();
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e)); float ms; CK(cudaEventElapsedTime(&ms, s, e)); t.push_back(ms / TIMED);
    }
    CK(cudaGetLastError());
    double m = 0, v = 0; for (int r = 1; r < REPS; ++r) m += t[r]; m /= (REPS - 1);
    for (int r = 1; r < REPS; ++r) v += (t[r] - m) * (t[r] - m);
    printf("ppcg %10.5f ms (+-%.2f%%) block %dx%dx%d grid %dx%d timed %d  err %.2e  %s\n", m, sqrt(v / (REPS - 2)) / m * 100, BX, BY, BZ, GX, GY, TIMED, err, err < 1e-3 ? "ok" : "WRONG");
    return 0;
}
