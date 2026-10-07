# 4-way comparison tables: sparse (naive), MDH, PPCG, cuSPARSE/cuBLAS

Three tables, 4 sizes each (512/1024/2048/4096). Non-square grids used for
512 (16x32) and 2048 (32x64) where a whole-number square Poisson grid
doesn't exist at that exact N.

| folder | table | 4th method | status |
|---|---|---|---|
| `matvec/` | matrix-vector (`Ap=A*p`) | cuSPARSE | done, all correct |
| `full_cg/` | full CG solve to convergence | cuSPARSE | done, all correct |
| `matmul/` | dense matrix-matrix (`S=A*B`) | cuBLAS (cuSPARSE doesn't do dense GEMM) | done, all correct |

Each folder has its own `results.md` with the full table, methodology, and
discussion. Rebuild any table with `./build_and_run_sweep.sh` inside that
folder (regenerates PPCG kernels via the real `ppcg` binary, recompiles MDH
via `-D` macros only -- no MDH regeneration needed per size).

## Headline findings across all three

- **matvec & full CG**: MDH fastest or tied-fastest at every size, PPCG
  close behind, cuSPARSE slowest on the isolated kernel (per-call overhead
  dominates at these still-small sizes) but competitive once folded into
  the full solver loop.
- **matmul**: the opposite story. cuBLAS wins overwhelmingly (10-15x), and
  *naive beats both MDH and PPCG* at every size -- because both generated
  GEMM kernels are running an untuned first-guess config here, not a
  searched/optimal one (see `matmul/results.md` for the caveat and what a
  tuned config can reach instead).
- **One reproducible float32 quirk**: PPCG converges in 87 iterations vs
  86 for everyone else at N=1024 in the full-CG table -- different
  summation order, not a correctness bug (see `full_cg/results.md`).

## What these tables establish

Sparse (CSR) and matrix-free implementations across 4 sizes for three
different operations, compared against vendor-library baselines
(cuSPARSE for sparse, cuBLAS for dense) rather than a mismatched one. See
the rest of this project (`5_3d_extension/`, `6_multigrid/`,
`7_preconditioning/`, `8_real_application/`, `roofline/`,
`scaling_analysis/`) for everything built on top of this foundation.
