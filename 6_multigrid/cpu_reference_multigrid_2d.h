// CPU reference for the 2D geometric multigrid V-cycle building blocks:
// Jacobi smoothing, residual, full-weighting restriction, bilinear
// prolongation. Correctness oracle for every GPU kernel in this folder --
// each is checked element-wise against this before its timing is trusted,
// same discipline as every other solver in this project.
//
// Convention: PROPERLY h-scaled operator at every level (standard textbook
// multigrid form) -- see the comment in test_cpu_vcycle.cu for why this
// differs from the CG solvers' "unscaled stencil" convention elsewhere in
// this project.
//
// Coarsening alignment: coarse point I corresponds to FINE index 2I+1 (not
// 2I). This is the standard textbook choice (Briggs et al., "A Multigrid
// Tutorial") and matters more than it looks: an earlier attempt using
// "coarse I <-> fine 2I" diverged once a 3rd coarsening level was added,
// traced to a bug -- that alignment puts the coarsest grid's first
// interior point at a DIFFERENT physical distance from the domain boundary
// than the coarse operator's own uniform-spacing assumption implies, so
// the coarse-grid correction equation silently solves the wrong boundary
// value problem. The "2I+1" alignment keeps every level's boundary
// distance consistent with its own grid spacing by construction. This
// requires grid sizes of the form m = 2^k - 1 (interior count) so
// m_fine = 2*m_coarse + 1 exactly at every level: 31, 63, 127, 255, ...
// down to a coarsest grid of 1 point (solvable exactly in closed form).

#ifndef CPU_REFERENCE_MULTIGRID_2D_H
#define CPU_REFERENCE_MULTIGRID_2D_H

#include <cstring>
#include <cstdlib>

static inline float at(const float* g, int m, int i, int j) {
    if (i < 0 || i >= m || j < 0 || j >= m) return 0.0f; // homogeneous Dirichlet oob
    return g[i * m + j];
}

// One damped Jacobi sweep for A_h[u] = b, A_h = (1/h^2) * (diag=4, neighbors=-1).
static void cpu_jacobi_smooth(int m, float h2, const float* b, float* u, float omega) {
    float* u_new = (float*)malloc(m * m * sizeof(float));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            float neighbor_sum = at(u, m, i - 1, j) + at(u, m, i + 1, j) +
                                  at(u, m, i, j - 1) + at(u, m, i, j + 1);
            float gs_update = (h2 * b[i * m + j] + neighbor_sum) / 4.0f;
            u_new[i * m + j] = (1.0f - omega) * u[i * m + j] + omega * gs_update;
        }
    }
    memcpy(u, u_new, m * m * sizeof(float));
    free(u_new);
}

// r = b - A_h[u]
static void cpu_residual(int m, float h2, const float* b, const float* u, float* r) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            float Au = (4.0f * u[i * m + j]
                     - at(u, m, i - 1, j) - at(u, m, i + 1, j)
                     - at(u, m, i, j - 1) - at(u, m, i, j + 1)) / h2;
            r[i * m + j] = b[i * m + j] - Au;
        }
    }
}

// Full-weighting restriction: fine (m_f x m_f) -> coarse (m_c x m_c),
// m_f = 2*m_c + 1. Coarse point (I,J) <-> fine point (2I+1, 2J+1).
static void cpu_restrict(int m_f, const float* r_f, float* r_c) {
    int m_c = (m_f - 1) / 2;
    for (int I = 0; I < m_c; ++I) {
        for (int J = 0; J < m_c; ++J) {
            int i = 2 * I + 1, j = 2 * J + 1;
            float center = at(r_f, m_f, i, j);
            float edges = at(r_f, m_f, i - 1, j) + at(r_f, m_f, i + 1, j) +
                          at(r_f, m_f, i, j - 1) + at(r_f, m_f, i, j + 1);
            float corners = at(r_f, m_f, i - 1, j - 1) + at(r_f, m_f, i - 1, j + 1) +
                            at(r_f, m_f, i + 1, j - 1) + at(r_f, m_f, i + 1, j + 1);
            r_c[I * m_c + J] = (4.0f * center + 2.0f * edges + corners) / 16.0f;
        }
    }
}

static inline float coarse_at(const float* e_c, int m_c, int I, int J) {
    if (I < 0 || I >= m_c || J < 0 || J >= m_c) return 0.0f;
    return e_c[I * m_c + J];
}

// Bilinear prolongation: coarse (m_c x m_c) -> fine (m_f x m_f),
// m_f = 2*m_c + 1, same alignment as cpu_restrict. Adds the interpolated
// correction directly onto the fine-grid array.
static void cpu_prolong_add(int m_c, const float* e_c, float* e_f) {
    int m_f = 2 * m_c + 1;
    for (int i = 0; i < m_f; ++i) {
        bool i_odd = (i % 2) != 0;
        int Ia = i_odd ? (i - 1) / 2 : (i / 2 - 1);
        int Ib = i_odd ? (i - 1) / 2 : (i / 2);
        for (int j = 0; j < m_f; ++j) {
            bool j_odd = (j % 2) != 0;
            int Ja = j_odd ? (j - 1) / 2 : (j / 2 - 1);
            int Jb = j_odd ? (j - 1) / 2 : (j / 2);
            float val;
            if (i_odd && j_odd) {
                val = coarse_at(e_c, m_c, Ia, Ja);
            } else if (!i_odd && j_odd) {
                val = 0.5f * (coarse_at(e_c, m_c, Ia, Ja) + coarse_at(e_c, m_c, Ib, Ja));
            } else if (i_odd && !j_odd) {
                val = 0.5f * (coarse_at(e_c, m_c, Ia, Ja) + coarse_at(e_c, m_c, Ia, Jb));
            } else {
                val = 0.25f * (coarse_at(e_c, m_c, Ia, Ja) + coarse_at(e_c, m_c, Ia, Jb) +
                                coarse_at(e_c, m_c, Ib, Ja) + coarse_at(e_c, m_c, Ib, Jb));
            }
            e_f[i * m_f + j] += val;
        }
    }
}

#endif
