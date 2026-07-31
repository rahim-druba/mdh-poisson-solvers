# Measured roofline profiling - Nsight Compute

Real, measured bandwidth and compute utilization numbers for all four
matvec kernels, instead of relying on a theoretical ridge-point estimate.
Profiled with Nsight Compute 2022.2.1 (`ncu`), one steady-state matvec
kernel launch (5th CG iteration, well past any cold-start effect) from
each of the four already-verified 2D solvers at N=4096 - the exact same
binaries used for every other table in this project, not separate
benchmark code.

**Hardware note**: this is measured on the RTX 3050 Laptop GPU used
throughout this project. An old_trial (dense storage) once assumed a
theoretical peak bandwidth figure for a different card entirely, so that
number isn't comparable here. ncu's own peak-bandwidth calibration on this
GPU comes out to ~150-160 GB/s (implied by `dram__bytes.sum.per_second`
divided by `dram__throughput.avg.pct_of_peak_sustained_elapsed`,
consistent across all four kernels independently - a good cross-check that
the percentages below are trustworthy).

## Results

| method | kernel duration | DRAM bytes moved | achieved BW | % of peak BW | % of peak compute |
|---|---:|---:|---:|---:|---:|
| sparse (CSR) | 4.83 us | 194.69 KB | 40.29 GB/s | 25.3% | 4.9% |
| mdh (matrix-free) | 2.88 us | 16.51 KB | 5.73 GB/s | 3.8% | 6.2% |
| ppcg (matrix-free) | 3.26 us | 17.66 KB | 5.41 GB/s | 3.7% | 2.9% |
| cusparse (CSR) | 6.94 us | 197.38 KB | 28.42 GB/s | 17.6% | 10.6% |

(Raw `ncu` reports: `sparse_ncu.txt`, `mdh_ncu.txt`, `ppcg_ncu.txt`,
`cusparse_ncu.txt` in this folder.)

## What this actually shows

**Two honest findings, one expected and one that complicates the simple
story:**

1. **Matrix-free genuinely moves ~11-12x less memory traffic than CSR, measured, not estimated.** MDH and PPCG move 16.5-17.7 KB; sparse and cuSPARSE (both true CSR, storing `val`/`col_idx` arrays) move 194.7-197.4 KB - an 11.2x-11.8x difference. This matches the expected mechanism: CSR requires loading both a value and an integer index for every nonzero, while matrix-free computes neighbor indices arithmetically and skips that indirect memory traffic entirely - now backed by a real measurement instead of an estimate.

2. **None of these kernels are anywhere close to saturating either the memory or compute roofline at N=4096.** The highest achieved is sparse at 25.3% of peak bandwidth; matrix-free MDH/PPCG sit at under 4%. Compute utilization tops out at 10.6%. This isn't a bug or a weak implementation - it directly confirms what this project's own `tables/matvec/results.md` already said qualitatively ("kernel launch and dispatch overhead dominates over compute at every size tested here"), now with hard numbers: **at N=4096, none of these kernels are actually bandwidth-bound or compute-bound in practice - they're overhead-bound.** A theoretical roofline argument based on arithmetic intensity alone describes the *asymptotic* regime; it doesn't describe what's actually happening at this problem size. That gap between theory and measured behavior is worth being upfront about.

## What this means going forward

- The matrix-free memory-traffic reduction is real and substantial (11-12x, measured), and meaningfully engaging the bandwidth roofline needs a larger problem size than what's tested here - consistent with `tables/matvec/results.md`'s own observation that the real bandwidth-bound crossover needs a substantially larger size than what's in that sweep.
- The natural follow-up, done separately: repeat this same profiling pass at the larger 3D sizes already built in `../5_3d_extension/` (up to N=32768) to see whether bandwidth utilization climbs meaningfully at larger N - see `results_3d.md`.
