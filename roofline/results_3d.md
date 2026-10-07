# Measured roofline, re-profiled at 3D sizes (follow-up)

> Note (revision): statements below about a compute-bound regime of the matrix-free kernel and about PPCG's single-block schedule refer to the untuned MDH
> configuration and PPCG's default schedule. See the update at the top of the root README for the tuned results.

Follow-up to `results.md`'s 2D N=4096 pass, which found none of the four
kernels were close to saturating either roofline ceiling and flagged
"worth re-profiling at larger N to see whether bandwidth utilization
actually climbs." It does - cleanly, and the four methods diverge once N is large enough to see it.

Profiled the same way (Nsight Compute, one steady-state kernel launch,
5th iteration), using the already-built, already-verified 3D solvers in
`../5_3d_extension/` at all four swept sizes (N=512/4096/13824/32768).
Raw reports: `{sparse,mdh,ppcg,cusparse}_3d_{8,16,24,32}_ncu.txt`.

## Results: % of peak memory bandwidth achieved

| N | sparse (CSR) | cusparse (CSR) | mdh (matrix-free) | ppcg (matrix-free) |
|---|---:|---:|---:|---:|
| 512 | 4.2% | 2.9% | 0.5% | 0.7% |
| 4096 | 28.3% | 21.5% | 3.4% | 1.5% |
| 13824 | 56.3% | 47.0% | 9.6% | 2.2% |
| 32768 | **72.9%** | **57.4%** | 14.2% | 2.3% |

(Implied GPU peak bandwidth, cross-checked from the N=32768 sparse
measurement: `113.55 / 0.7293 ≈ 156 GB/s`, consistent with the 2D pass's
~150-160 GB/s and with every other size/method's own implied peak.)

## Results: % of peak compute throughput achieved

| N | sparse | cusparse | mdh | ppcg |
|---|---:|---:|---:|---:|
| 512 | 0.9% | 2.1% | 1.2% | 0.4% |
| 4096 | 5.6% | 13.7% | 9.6% | 1.0% |
| 13824 | 11.0% | 29.2% | 28.0% | 1.7% |
| 32768 | 14.0% | 35.4% | **40.0%** | 1.9% |

## What this shows - three different behaviors, one per method family

1. **CSR-based methods (sparse, cuSPARSE) do become memory-bound as
   N grows** - this is the clean confirmation the 2D N=4096 pass couldn't
   show. Sparse climbs from 4.2% to 72.9% of peak bandwidth, a monotonic,
   near-linear climb toward saturation; cuSPARSE follows the same shape
   (2.9% -> 57.4%). At N=32768, sparse is close to the memory
   roofline ceiling. **This is the measured evidence a theoretical
   ridge-point argument alone can't provide** - it's not automatically
   true at every size, but it does become true, measurably, once the
   problem is big enough.

2. **MDH's matrix-free kernel does *not* follow the same path** - its
   bandwidth utilization also climbs (0.5% -> 14.2%) but stays far below
   the CSR methods throughout, while its **compute** utilization climbs
   much faster (1.2% -> 40.0%, becoming the single highest compute
   utilization of any method at the largest size). This is a real,
   distinct signature: matrix-free recomputes neighbor offsets and
   coefficients arithmetically instead of reading them from memory, so as
   N grows it becomes increasingly **compute-bound**, not memory-bound - a
   different roofline regime than CSR, not just a faster version of the
   same one.

3. **PPCG stays flat and low at every size** (0.7% -> 2.3% bandwidth,
   0.4% -> 1.9% compute) - independent confirmation, via a third
   measurement method now (after `cudaEvent` timing and the observed
   `block(4,4,M)`/`grid(1,1)` launch config), that PPCG's 3D schedule
   never engages more than one SM regardless of problem size. It isn't
   becoming memory- or compute-bound because it's neither - it's
   structurally underutilizing the GPU at every size tested.

## What this means

The 2D-only pass in `results.md` was correct but incomplete - it showed the
memory-bound claim wasn't demonstrated at N=4096, without showing whether
it would be true at any size. This pass closes that gap: the claim **is**
demonstrably true, but only once N is large enough (roughly N>10000 in
this problem, based on where sparse/cuSPARSE cross 50% of peak bandwidth)
- and even then, it's specifically true for CSR-based implementations.
Matrix-free's roofline profile is different (compute-bound
at scale, not memory-bound), which is a more specific result than a simple "confirms memory-bound" claim, since it means MDH's speed advantage
over CSR isn't just "less memory traffic" in an unqualified sense - it's a
real regime shift.
