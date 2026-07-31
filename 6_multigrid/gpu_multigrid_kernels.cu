// GPU multigrid kernels: damped Jacobi smoothing, residual, full-weighting
// restriction, bilinear prolongation. Direct one-thread-per-grid-point port
// of the verified CPU reference in cpu_reference_multigrid_2d.h -- same
// formulas, same "coarse I <-> fine 2I+1" alignment, same h^2-scaled
// operator convention. Grid family m = 2^k - 1 (see that file's header
// comment for why this alignment matters).

#ifndef GPU_MULTIGRID_KERNELS_CU
#define GPU_MULTIGRID_KERNELS_CU

#include <cuda_runtime.h>

__device__ __forceinline__ float dev_at(const float* g, int m, int i, int j) {
    if (i < 0 || i >= m || j < 0 || j >= m) return 0.0f;
    return g[i * m + j];
}

__device__ __forceinline__ float dev_coarse_at(const float* e_c, int m_c, int I, int J) {
    if (I < 0 || I >= m_c || J < 0 || J >= m_c) return 0.0f;
    return e_c[I * m_c + J];
}

// One damped Jacobi sweep, ping-pong buffered (reads u_old, writes u_new).
__global__ void jacobi_smooth_kernel(int m, float h2, const float* __restrict__ b,
                                      const float* __restrict__ u_old,
                                      float* __restrict__ u_new, float omega) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= m) return;
    float neighbor_sum = dev_at(u_old, m, i - 1, j) + dev_at(u_old, m, i + 1, j) +
                          dev_at(u_old, m, i, j - 1) + dev_at(u_old, m, i, j + 1);
    float gs_update = (h2 * b[i * m + j] + neighbor_sum) / 4.0f;
    u_new[i * m + j] = (1.0f - omega) * u_old[i * m + j] + omega * gs_update;
}

// r = b - A_h[u]
__global__ void residual_kernel(int m, float h2, const float* __restrict__ b,
                                 const float* __restrict__ u, float* __restrict__ r) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= m) return;
    float Au = (4.0f * u[i * m + j]
             - dev_at(u, m, i - 1, j) - dev_at(u, m, i + 1, j)
             - dev_at(u, m, i, j - 1) - dev_at(u, m, i, j + 1)) / h2;
    r[i * m + j] = b[i * m + j] - Au;
}

// Full-weighting restriction: fine (m_f x m_f) -> coarse (m_c x m_c), m_f = 2*m_c+1.
__global__ void restrict_kernel(int m_f, const float* __restrict__ r_f, float* __restrict__ r_c) {
    int m_c = (m_f - 1) / 2;
    int I = blockIdx.y * blockDim.y + threadIdx.y;
    int J = blockIdx.x * blockDim.x + threadIdx.x;
    if (I >= m_c || J >= m_c) return;
    int i = 2 * I + 1, j = 2 * J + 1;
    float center = dev_at(r_f, m_f, i, j);
    float edges = dev_at(r_f, m_f, i - 1, j) + dev_at(r_f, m_f, i + 1, j) +
                  dev_at(r_f, m_f, i, j - 1) + dev_at(r_f, m_f, i, j + 1);
    float corners = dev_at(r_f, m_f, i - 1, j - 1) + dev_at(r_f, m_f, i - 1, j + 1) +
                    dev_at(r_f, m_f, i + 1, j - 1) + dev_at(r_f, m_f, i + 1, j + 1);
    r_c[I * m_c + J] = (4.0f * center + 2.0f * edges + corners) / 16.0f;
}

// Bilinear prolongation: coarse (m_c x m_c) -> fine (m_f x m_f), m_f = 2*m_c+1.
// Adds the interpolated correction directly onto the fine-grid array.
__global__ void prolong_add_kernel(int m_c, const float* __restrict__ e_c, float* __restrict__ e_f) {
    int m_f = 2 * m_c + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m_f || j >= m_f) return;

    bool i_odd = (i % 2) != 0, j_odd = (j % 2) != 0;
    int Ia = i_odd ? (i - 1) / 2 : (i / 2 - 1);
    int Ib = i_odd ? (i - 1) / 2 : (i / 2);
    int Ja = j_odd ? (j - 1) / 2 : (j / 2 - 1);
    int Jb = j_odd ? (j - 1) / 2 : (j / 2);

    float val;
    if (i_odd && j_odd) {
        val = dev_coarse_at(e_c, m_c, Ia, Ja);
    } else if (!i_odd && j_odd) {
        val = 0.5f * (dev_coarse_at(e_c, m_c, Ia, Ja) + dev_coarse_at(e_c, m_c, Ib, Ja));
    } else if (i_odd && !j_odd) {
        val = 0.5f * (dev_coarse_at(e_c, m_c, Ia, Ja) + dev_coarse_at(e_c, m_c, Ia, Jb));
    } else {
        val = 0.25f * (dev_coarse_at(e_c, m_c, Ia, Ja) + dev_coarse_at(e_c, m_c, Ia, Jb) +
                        dev_coarse_at(e_c, m_c, Ib, Ja) + dev_coarse_at(e_c, m_c, Ib, Jb));
    }
    e_f[i * m_f + j] += val;
}

#endif
