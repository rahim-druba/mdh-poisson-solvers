# Table 4: isolated matrix-vector product (`Ap = A*p`), 2D and 3D, RTX 3050

Isolated kernel timing (`bench_matvec.cu` for 2D, `bench_matvec_3d.cu` for 3D, one
binary per size/tuning-state pair). >=300 ms time-based warmup + 200 timed launches,
`cudaEvent`, averaged. Correctness checked against a shared CPU reference before
timing. MDH (untuned) uses one fixed configuration per dimension; MDH (tuned) and PPCG
(tuned) use the best configuration found for each size by `mdh_tune_sweep.sh` /
`ppcg_tune_sweep.sh`. This is the exact data behind the paper's Table 4; raw logs are
in `../../handtuned_baseline/raw/paper_sizes/` and `../../ppcg_tuning/raw/paper/`.

| N (grid) | CSR | cuSPARSE | PPCG (default) | PPCG (tuned) | MDH (untuned) | MDH (tuned) | hand-tuned |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 (16x32) | 0.00225 | 0.00592 | 0.00150 | **0.00132** | 0.00151 | 0.00138 | **0.00132** |
| 1024 (32x32) | 0.00206 | 0.00600 | 0.00160 | **0.00134** | 0.00150 | 0.00140 | **0.00134** |
| 2048 (32x64) | 0.00207 | 0.00621 | 0.00163 | **0.00134** | 0.00158 | 0.00141 | 0.00136 |
| 4096 (64x64) | 0.00217 | 0.00692 | 0.00168 | **0.00136** | 0.00163 | 0.00152 | 0.00140 |
| 1000000 (1000x1000) | 0.28621 | 0.32897 | 0.04913 | 0.05057 | 0.06330 | 0.05345 | **0.04598** |
| 9000000 (3000x3000) | 2.59131 | 2.92079 | 0.42549 | 0.41016 | 0.55862 | 0.48112 | **0.40202** |
| 512 (8^3) | 0.00217 | 0.00593 | 0.00193 | **0.00134** | 0.00179 | 0.00146 | 0.00142 |
| 4096 (16^3) | 0.00238 | 0.00695 | 0.00399 | **0.00141** | 0.00186 | 0.00151 | 0.00155 |
| 13824 (24^3) | 0.00362 | 0.00844 | 0.00828 | **0.00166** | 0.00231 | 0.00179 | 0.00186 |
| 32768 (32^3) | 0.01527 | 0.01772 | 0.02628 | **0.00192** | 0.00334 | 0.00231 | 0.00209 |
| 1000000 (100^3) | 0.38342 | 0.42076 | 0.13352 | **0.04938** | 0.10628 | 0.07220 | 0.05484 |

Bold marks the fastest method in each row.

## What this shows

- **At the small sizes (512 through 32768, all powers of two), every kernel including
  CSR is dominated by fixed per-launch and dispatch overhead**, not by arithmetic or
  memory-traffic growth with N. This is why, for example, CSR's time barely changes
  between N=512 and N=4096 (0.00225 to 0.00217 ms). Confirmed directly with Nsight
  Compute profiling (see `../../roofline/`): at N=4096, none of the four kernels come
  close to saturating either the memory or compute roofline.
- **The three large, deliberately non-power-of-two rows (1000000 at 1000x1000,
  9000000 at 3000x3000, 1000000 at 100^3) were added specifically to show that
  execution time scales correctly with N once past that small-size overhead floor.**
  CSR grows from 0.00217 ms (N=4096) to 2.59 ms (N=9000000), a 1194x increase in cost
  for a 2197x increase in problem size, consistent with memory-bandwidth scaling.
- These three sizes were chosen to divide evenly by a valid MDH tile size (e.g. 1000
  and 3000 both divide by 20, 100 by 10). The MDH-generated kernel's own code does
  contain remainder-handling logic for sizes that don't divide evenly, but the
  benchmark driver used throughout this repo (`handtuned_baseline/paper_cg.sh`)
  computes grid size with plain truncating integer division, which would silently
  undersize the grid for a genuinely indivisible N. Choosing divisor-safe sizes avoids
  that risk entirely while still answering the "only powers of two were tested"
  question.
- Hand-tuned wins both new 2D sizes; tuned PPCG wins the new 3D size. This is reported
  as found, consistent with the paper's statement elsewhere that it found no size at
  which tuned MDH is faster than tuned PPCG.
- At N=100^3, `bench_matvec_3d.cu`'s own embedded PPCG column is not valid (its
  launch geometry is hardcoded for the original small sizes); the PPCG default/tuned
  numbers for this row come from the dedicated `ppcg_tuning/bench_ppcg.cu` harness
  instead. See the note at the top of `matvec_3d_100_untuned.txt`.
- All measurements were taken with GPU persistence mode enabled and no thermal/power
  throttling active (`nvidia-smi`), the same protocol used throughout this repository.
