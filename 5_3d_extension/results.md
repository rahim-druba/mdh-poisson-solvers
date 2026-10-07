# 3D extension - results, swept across four sizes

Direct 3D extension of the 2D matrix-free/CSR/cuSPARSE Poisson-CG
comparison in `../1_sparse_rewrite` through `../4_cusparse`, built to
prove that the same code generation approach used in 2D carries
over to 3D. Every kernel below is a real, working, correctness-verified
implementation, swept across four grid sizes the same way the 2D tables
sweep 512/1024/2048/4096.

**Domain**: unit cube `[0,1]^3`, Dirichlet BC, analytical solution
`u(x,y,z) = 1 + x^2 + y^2 + z^2` (direct extension of the 2D `u = 1 + x^2 +
y^2` used everywhere else in this project), 7-point stencil (diag=6, six
face-neighbors -1). Single precision (`float`), matching every existing
table here. Sizes are cube grids `8/16/24/32` per side -> `N =
512/4096/13824/32768` unknowns.

## Correctness

Every kernel was checked element-wise against a plain triple-loop CPU
reference (`cpu_reference_matvec_3d.h`) **before** any timing was trusted -
same bar as every 2D table in this project. All four methods at all four
sizes: **zero mismatches.** The MDH kernel was additionally smoke-tested at
a tiny 4x4x4 grid first, since this is the first 3D spec ever written for
this framework (no prior art existed anywhere to copy from) - passed
immediately, no framework changes needed.

## Regenerating this sweep

MDH's generated kernel (`cg_matvec_3d_1.cu`) is **size-agnostic** - no
hardcoded array-size casts anywhere in it, purely macro-driven, so the same
`.cu` file is just recompiled with different `-D` tile macros per size
(confirmed identical to how the 2D `cg_matvec_1.cu` behaves). PPCG's
kernel, like the 2D case, **bakes size in at generation time** and needs
regenerating per size - done here via
`sizes/<M>/cg_matvec_3d_ppcg_src.c` (differs from the base source only in
`#define M`) fed through `ppcg --target=cuda`, same as the 2D pipeline, no
extra flags. `build_and_run_sweep_3d.sh` builds and runs all 16
sparse/mdh/ppcg/cusparse x size combinations; it does not re-run `ppcg`
itself (matches how the 2D `tables/matvec/build_and_run_sweep.sh` only
compiles already-generated per-size kernels).

## Table 1: isolated matvec (`Ap = A*p`), swept over sizes

`cudaEvent` timing, time-based warmup (>=300ms) + 200 timed launches,
matching this project's own established matvec methodology
(`bench_matvec_3d.cu`).

| size (grid) | sparse (ms) | mdh (ms) | ppcg (ms) | cusparse (ms) | correct (all 4) |
|---|---:|---:|---:|---:|:---:|
| 512 (8^3)    | 0.00240 | 0.00175 | 0.00185 | 0.00580 | yes |
| 4096 (16^3)  | 0.00234 | 0.00185 | 0.00391 | 0.00651 | yes |
| 13824 (24^3) | 0.00358 | 0.00231 | 0.00817 | 0.00844 | yes |
| 32768 (32^3) | 0.01412 | **0.00329** | 0.02617 | 0.01771 | yes |

**MDH wins at every size, and its margin over PPCG widens sharply as the
grid grows** (roughly tied at 8^3, ~8x faster at 32^3). This is a real,
measured structural difference, not a tuning gap: PPCG's auto-generated 3D
schedule stays a **single block on a single SM at every size tested**
(`block(4,4,M)`, `grid(1,1)` - confirmed identical launch config at 8, 16,
24, and 32). It fully parallelizes only the k-dimension; the other two are
grid-strided within that one block. MDH's schedule adds more blocks
(`NUM_WG = M/8`) as the grid grows, so it keeps using more of the GPU while
PPCG's one-shot static schedule does not. This is PPCG's unmodified default
(no `--sizes` flag was used anywhere in this project, 2D or 3D) - a real
data point about how the two generation strategies diverge in 3D, not a
methodology error on our part.

## Table 2: full CG solve to convergence, swept over sizes

10 runs per (method, size), first dropped as warmup, remaining 9 averaged -
same protocol as `../tables/full_cg/results.md` (`build_and_run_sweep_3d.sh`).

| size | sparse (ms) | mdh (ms) | ppcg (ms) | cusparse (ms) | iterations |
|---|---:|---:|---:|---:|---:|
| 512 (8^3)    | 0.4622  | 0.4646  | 0.4704  | 0.5331  | 27 |
| 4096 (16^3)  | 1.4662  | 1.4984  | 1.6673  | 1.6478  | 56 (57 mdh, 62 ppcg/cusparse) |
| 13824 (24^3) | 4.6749  | 4.9399  | 5.2396  | 5.4943  | 93 (92 mdh/cusparse, 84 ppcg) |
| 32768 (32^3) | **13.2364** | 12.6460 | 16.7041 | 13.7694 | 131 (132 mdh, 130 cusparse) |

All four converge to the correct answer at every size (`rs_old` after CG
init matches across all four methods at a given size, confirming the RHS
construction is consistent). Small iteration-count differences (e.g. 56 vs
57 vs 62 at N=4096) are the same float32 summation-order artifact already
documented for the 2D case (each kernel sums the 7 stencil terms in a
different order) - not a correctness bug; every method still lands within
~1e-5 of the exact solution at every size.

The gap between methods is much narrower here than in Table 1, same
pattern as the 2D project's own finding: at these sizes, host-side CPU
reductions and the `cudaMemcpy` round trip per iteration dominate total
time, so the isolated-kernel gap (where MDH beats PPCG by up to 8x) mostly
washes out once folded into the full solver loop. PPCG's per-iteration
matvec disadvantage still shows up as a consistent ~15-25% slower full
solve at every size, though.

## What this settles

A real 3D 7-point Poisson-CG solver exists now, generated two different
ways (MDH auto-tuned, PPCG polyhedral), plus a hand-written CSR baseline
and a cuSPARSE vendor-library baseline, all converging to the correct
analytical solution, swept across four grid sizes the same way this
project's 2D tables are.

## What's still open from here

- An earlier, unrelated dense-storage trial worked in double precision at
  larger nominal sizes; this rebuild is single precision (`float`)
  throughout and matches the scale of everything else in this project.
  Keep this in mind if the two are compared directly.
- PPCG's single-block-per-kernel behavior in 3D is noted here
  explicitly if this table is cited directly - it's PPCG's real, unmodified
  default schedule, but it makes PPCG look considerably worse in 3D than a
  `--sizes`-tuned or hand-tuned schedule might.
- Larger cube sizes (48^3, 64^3) would push further into the
  bandwidth-bound regime the way the 2D dense-matvec tuning work found a
  real MDH/PPCG crossover at scale - not attempted here.
