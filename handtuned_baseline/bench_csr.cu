// CSR-scalar and cuSPARSE SpMV for the same Poisson operator, large sizes (RTX 5090 run, reviewer issues #1/#4).
// Same operator, input (srand(42)), CPU oracle and cudaEvent protocol as bench_handtuned.cu / the paper's bench_matvec.
// Build: -DDIM=2 -DROWS=R -DCOLS=C   or   -DDIM=3 -DSIDE=M       Prints one line per method.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>
#define CK(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) { fprintf(stderr, "CUDA '%s' at %d\n", cudaGetErrorString(_e), __LINE__); exit(1); } } while (0)
#define CS(e) do { cusparseStatus_t _s = (e); if (_s != CUSPARSE_STATUS_SUCCESS) { fprintf(stderr, "cuSPARSE %d at %d\n", (int)_s, __LINE__); exit(1); } } while (0)
#ifndef REPS
#define REPS 10
#endif
static const int TIMED = 200; static const float MIN_WARMUP_MS = 300.0f;

__global__ void spmv_csr_kernel(int n, const int* __restrict__ rp, const int* __restrict__ ci,
                                const float* __restrict__ v, const float* __restrict__ p, float* __restrict__ ap) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) { float s = 0.f; for (int k = rp[row]; k < rp[row + 1]; ++k) s += v[k] * p[ci[k]]; ap[row] = s; }
}

template <class F> static double timeit(F launch) {
    cudaEvent_t ws, we, s, e; CK(cudaEventCreate(&ws)); CK(cudaEventCreate(&we)); CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    CK(cudaEventRecord(ws)); float w = 0;
    do { launch(); CK(cudaEventRecord(we)); CK(cudaEventSynchronize(we)); CK(cudaEventElapsedTime(&w, ws, we)); } while (w < MIN_WARMUP_MS);
    std::vector<double> t;
    for (int r = 0; r < REPS; ++r) {
        CK(cudaEventRecord(s)); for (int i = 0; i < TIMED; ++i) launch();
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e)); float ms; CK(cudaEventElapsedTime(&ms, s, e)); t.push_back(ms / TIMED);
    }
    CK(cudaGetLastError());
    double m = 0; for (int r = 1; r < REPS; ++r) m += t[r];
    return m / (REPS - 1);
}

int main() {
#if DIM == 2
    const long R = ROWS, C = COLS, N = R * C; const int nb = 4;
    printf("=== CSR / cuSPARSE: 2D %ldx%ld, N=%ld ===\n", R, C, N);
#else
    const long M = SIDE, N = M * M * M; const int nb = 6;
    printf("=== CSR / cuSPARSE: 3D side %ld, N=%ld ===\n", M, N);
#endif
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); printf("device: %s\n", pr.name);
    std::vector<float> p(N), ref(N), res(N);
    srand(42); for (long i = 0; i < N; ++i) p[i] = (float)(rand() % 100) / 10.0f - 5.0f;
    std::vector<int> rp(N + 1); std::vector<int> ci; std::vector<float> va;
    ci.reserve(N * (nb / 2 + 1) + N); va.reserve(ci.capacity());
    rp[0] = 0;
    for (long i = 0; i < N; ++i) {
        float s = (float)nb * p[i];
        auto add = [&](long j, bool ok) { if (ok) { ci.push_back((int)j); va.push_back(-1.0f); s -= p[j]; } };
#if DIM == 2
        long a = i / C, b = i % C;
        add(i - C, a > 0); add(i - 1, b > 0);
        ci.push_back((int)i); va.push_back((float)nb);
        add(i + 1, b < C - 1); add(i + C, a < R - 1);
#else
        long a = i / (M * M), b = (i / M) % M, c = i % M;
        add(i - M * M, a > 0); add(i - M, b > 0); add(i - 1, c > 0);
        ci.push_back((int)i); va.push_back((float)nb);
        add(i + 1, c < M - 1); add(i + M, b < M - 1); add(i + M * M, a < M - 1);
#endif
        ref[i] = s; rp[i + 1] = (int)ci.size();
    }
    long nnz = (long)ci.size();
    int *d_rp, *d_ci; float *d_v, *d_p, *d_ap;
    CK(cudaMalloc(&d_rp, (N + 1) * 4)); CK(cudaMalloc(&d_ci, nnz * 4)); CK(cudaMalloc(&d_v, nnz * 4));
    CK(cudaMalloc(&d_p, N * 4)); CK(cudaMalloc(&d_ap, N * 4));
    CK(cudaMemcpy(d_rp, rp.data(), (N + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ci, ci.data(), nnz * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v, va.data(), nnz * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_p, p.data(), N * 4, cudaMemcpyHostToDevice));
    auto check = [&](const char* name, double ms) {
        CK(cudaMemcpy(res.data(), d_ap, N * 4, cudaMemcpyDeviceToHost));
        double err = 0; for (long i = 0; i < N; ++i) err = fmax(err, fabs((double)res[i] - ref[i]));
        printf("  %-12s %10.5f ms   err %.2e  %s\n", name, ms, err, err < 1e-3 ? "ok" : "WRONG");
    };
    int n = (int)N; dim3 nb256(256), ng((n + 255) / 256);
    CK(cudaMemset(d_ap, 0, N * 4));
    double t1 = timeit([&] { spmv_csr_kernel<<<ng, nb256>>>(n, d_rp, d_ci, d_v, d_p, d_ap); });
    check("csr-scalar", t1);
    cusparseHandle_t h; CS(cusparseCreate(&h));
    cusparseSpMatDescr_t A; CS(cusparseCreateCsr(&A, N, N, nnz, d_rp, d_ci, d_v, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnVecDescr_t vp, vap; CS(cusparseCreateDnVec(&vp, N, d_p, CUDA_R_32F)); CS(cusparseCreateDnVec(&vap, N, d_ap, CUDA_R_32F));
    const float al = 1.f, be = 0.f; size_t bs = 0;
    CS(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &al, A, vp, &be, vap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bs));
    void* buf; CK(cudaMalloc(&buf, bs ? bs : 1));
    CK(cudaMemset(d_ap, 0, N * 4));
    double t2 = timeit([&] { CS(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &al, A, vp, &be, vap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf)); });
    check("cusparse", t2);
    return 0;
}
