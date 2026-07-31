# Table 1: matrix-vector operation (`Ap = A*p`), 4-way, size sweep

Isolated kernel timing (`bench_matvec.cu`, one binary per size). 20 warmup +
200 timed launches, `cudaEvent`, averaged. Correctness checked against a
shared CPU reference before timing. Non-square grids used for the two sizes
that aren't perfect squares: 512 = 16x32, 2048 = 32x64.

| size (grid)   | sparse (ms) | mdh (ms) | ppcg (ms) | cusparse (ms) | correct (all 4) |
|---------------|------------:|---------:|----------:|---------------:|:----------------:|
| 512 (16x32)   | 0.00239 | 0.00186 | 0.00178 | 0.00697 | yes |
| 1024 (32x32)  | 0.00228 | 0.00168 | 0.00176 | 0.00671 | yes |
| 2048 (32x64)  | 0.00230 | 0.00176 | 0.00180 | 0.00688 | yes |
| 4096 (64x64)  | 0.00240 | 0.00180 | 0.00183 | 0.00717 | yes |

## What this shows

- **MDH is fastest or tied-fastest at every size**, PPCG a close second,
  sparse (naive CSR) third, cuSPARSE consistently ~3-4x slower.
- **Timings barely move across the size range.** These are all still small
  problems for a GPU (max 4096 threads of actual work) - kernel launch and
  dispatch overhead dominates over compute at every size tested here, which
  is why cuSPARSE's fixed per-call overhead shows up as a constant ~3-4x
  penalty regardless of size, rather than shrinking as a fraction of total
  time the way it would at a much larger problem. The real bandwidth-bound
  crossover (where cuSPARSE and library maturity start to matter) needs a
  substantially larger size than what's in this sweep - see the roofline
  folder for that crossover measured directly at larger sizes.
- MDH's generated kernel needed **zero regeneration** across sizes - same
  `cg_matvec_1.cu`, just recompiled with different `-D` tile/grid macros.
  PPCG's kernel had to be **regenerated per size** (`ppcg --target=cuda`
  rerun on a size-specific `.c` source) since it bakes array strides as
  literal constants; each size picked its own launch config automatically
  (512: block 16x16/grid 1x1 ... 4096: block 16x32/grid 2x2).
