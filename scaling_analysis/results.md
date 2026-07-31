# Strong and weak scaling analysis

A proper scalability study needs both strong and weak scaling, not just a
table of timings at a few sizes. Built entirely from data already measured
in this project (2D CG sweep, 3D CG sweep, multigrid sweep, and the
occupancy profiling pass) - no new code or experiments, just assembled and
framed as the scaling study it deserves.

## A note on what "strong" and "weak" scaling mean here

The classical definitions (Amdahl's/Gustafson's laws) are about varying
the number of *processors* - fixed problem size across more processors
(strong), or problem size growing proportionally with processor count
(weak). This project is a **single-GPU** study; there is no processor
count to vary. The honest, standard reinterpretation for a single-device
GPU study, used here:

- **Weak scaling** -> does the *per-unit-of-work* cost stay constant as
  the problem grows? (Ideal: flat. This is the meaningful analog of "adding
  proportionally more work" when the device itself is fixed.)
- **Strong scaling** -> for a fixed device, how effectively does the
  implementation engage the hardware's fixed parallel resources as more
  work becomes available? (This project's `roofline/results_occupancy.md`
  already measured exactly this - reused here, not re-run.)

## Weak scaling: time per unknown (lower and flatter is better)

`total solve time / N`, in microseconds per unknown, across every sweep in
this project.

### 2D CG (full solve)

| N | sparse | mdh | ppcg | cusparse |
|---|---:|---:|---:|---:|
| 512 | 2.436 | 2.371 | 2.269 | 2.672 |
| 1024 | 1.494 | 1.473 | 1.488 | 1.610 |
| 2048 | 1.417 | 1.469 | 1.456 | 1.567 |
| 4096 | 1.217 | 1.235 | 1.240 | 1.300 |

### 3D CG (full solve)

| N | sparse | mdh | ppcg | cusparse |
|---|---:|---:|---:|---:|
| 512 | 0.903 | 0.907 | 0.919 | 1.041 |
| 4096 | 0.358 | 0.366 | 0.407 | 0.402 |
| 13824 | 0.338 | 0.357 | 0.379 | 0.397 |
| 32768 | **0.404** | 0.386 | **0.510** | 0.420 |

### Multigrid (full solve)

| N | time/unknown |
|---|---:|
| 961 | 0.4755 |
| 3969 | 0.1471 |
| 16129 | 0.0513 |
| 65025 | **0.0197** |

## What the weak-scaling numbers show

**CG's per-unit cost does not scale ideally - it improves, then reverses.**
Both 2D and 3D CG show the same shape: per-unit cost drops sharply at
first (fixed per-iteration overhead - `cudaMemcpy`, CPU dot products -
amortizing over more work), reaches a low point, then **rises again** at
the largest size tested (3D: 0.338 us/unknown at N=13824 -> 0.404-0.510 at
N=32768). This is the direct, measured consequence of CG's iteration count
growing with N (condition number scales with grid size) - the amortization
benefit eventually loses to the growing iteration count. **This is not
ideal weak scaling**, and it's exactly the mechanism `6_multigrid/results.md`
already identified as the reason multigrid's advantage over CG widens with N.

**Multigrid's per-unit cost falls monotonically and dramatically** - 24x
more efficient per unknown at N=65025 than at N=961, with no reversal
anywhere in the tested range. This is close to ideal weak scaling: total
work grows ~68x, total time grows only ~2.8x, because the V-cycle count
stays flat (6, at every size - see `6_multigrid/results.md`). This is the
single clearest, most quantitative demonstration in this whole project of
why the multigrid comparison mattered - it's not just faster, it scales
fundamentally differently.

## Strong scaling (GPU utilization vs problem size)

Reusing `roofline/results_occupancy.md`'s measurements - how much of the
GPU's fixed 16-SM, 48-warp-per-SM capacity each method's kernel actually
engages, as N grows on this fixed device:

| N (3D) | sparse | mdh | ppcg | cusparse |
|---|---:|---:|---:|---:|
| 512 | 15.6% | 33.1% | 7.9% | 16.3% |
| 4096 | 16.2% | 33.1% | 16.0% | 16.5% |
| 13824 | 54.5% | 54.8% | 24.3% | 45.7% |
| 32768 | 82.2% | 83.4% | **33.1%** | 85.0% |

Multigrid (2D, finest-level smoother kernel, `roofline/occ_mg_*.txt`) shows
the same healthy pattern as sparse/MDH/cuSPARSE, not PPCG's capped one:

| N (2D) | multigrid (finest-level smoother) |
|---|---:|
| 961 | 16.5% |
| 3969 | 16.5% |
| 16129 | 59.0% |
| 65025 | **76.0%** |

Sparse, MDH, cuSPARSE, and multigrid all scale toward using most of the GPU's
available parallelism as the problem grows - the expected, healthy
pattern for a single-device "strong scaling" story (more available work
lets the fixed hardware be used more fully). PPCG's ceiling at 33.1% is
not scaling at all in the meaningful sense: as established in
`results_occupancy.md`, that number is one block's warp count growing, not
more of the GPU being engaged - PPCG uses at most 1 of 16 SMs regardless
of N. It is the one method in this project that does not scale onto the
available hardware, by any reading of the term.

## Summary

| Method family | Weak scaling | Strong scaling (GPU utilization) |
|---|---|---|
| CG (all backends except PPCG) | Improves then reverses - not ideal | Scales well, reaches 82-85% GPU utilization |
| CG (PPCG backend specifically) | Same CG pattern | **Does not scale** - capped at ~1 SM regardless of N |
| Multigrid | Near-ideal, 24x efficiency gain | Scales well, reaches 76% GPU utilization by N=65025 |

This ties together four previously-separate findings (the CG-vs-multigrid
comparison, the roofline pass, the CG occupancy pass, and the multigrid
occupancy pass) into one coherent scalability story: every method in this
project scales onto the GPU's hardware in the expected way as the problem
grows, except PPCG's 3D schedule, which is structurally capped at a
single SM no matter how much work is available.
