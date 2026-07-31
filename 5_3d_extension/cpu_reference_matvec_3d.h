// CPU reference for the 3D 7-point Poisson stencil matvec: Ap = A*p.
// Correctness oracle for every GPU kernel in this folder (MDH, PPCG,
// hand-written CSR, cuSPARSE) -- each is checked element-wise against this
// before its timing numbers are trusted. Mirrors the 2D 5-point reference
// methodology used throughout ../1_sparse_rewrite .. ../4_cusparse.
//
// Interior grid is m x m x m (m = fulln - 2), unknowns flattened
// row-major as idx = (i*m + j)*m + k, i,j,k in [0, m). Dirichlet oob: a
// missing neighbor outside [0,m) contributes 0 (its value was already
// folded into the RHS b at matrix-construction time, same as the 2D case).

#ifndef CPU_REFERENCE_MATVEC_3D_H
#define CPU_REFERENCE_MATVEC_3D_H

static void cpu_matvec_3d(int m, const float* p, float* Ap) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < m; ++k) {
                int idx = (i * m + j) * m + k;
                float sum = 6.0f * p[idx];
                if (i > 0)     sum -= p[((i - 1) * m + j) * m + k];
                if (i < m - 1) sum -= p[((i + 1) * m + j) * m + k];
                if (j > 0)     sum -= p[(i * m + (j - 1)) * m + k];
                if (j < m - 1) sum -= p[(i * m + (j + 1)) * m + k];
                if (k > 0)     sum -= p[(i * m + j) * m + (k - 1)];
                if (k < m - 1) sum -= p[(i * m + j) * m + (k + 1)];
                Ap[idx] = sum;
            }
        }
    }
}

#endif
