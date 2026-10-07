# GPU occupancy profiling

GPU occupancy gives information that timing and bandwidth do not,
so it is measured directly. Profiled
with `ncu`'s built-in `Occupancy` section, same steady-state kernel
launches as the roofline passes, across all four 2D matvec kernels at
N=4096 and the full 3D sweep (N=512-32768). Raw reports: `occ_*.txt` in
this directory.

## Theoretical occupancy: 100% everywhere, for every kernel

Every single kernel/method/size combination profiled reports
**Theoretical Occupancy = 100%** - none of these kernels are limited by
register usage, shared memory, or block-size choice. This rules out the
"textbook" occupancy story (a kernel using too many registers or too much
shared memory per thread) entirely. Whatever's happening to *achieved*
occupancy is not a per-kernel resource-usage problem - it's a launch
configuration / grid-sizing problem.

## Achieved occupancy: driven almost entirely by grid size, not kernel design

3D sweep, achieved occupancy (% of the GPU's 48 max warps/SM kept busy):

| N | sparse (CSR) | mdh (matrix-free) | ppcg (matrix-free) | cusparse (CSR) |
|---|---:|---:|---:|---:|
| 512 | 15.6% | 33.1% | 7.9% | 16.3% |
| 4096 | 16.2% | 33.1% | 16.0% | 16.5% |
| 13824 | 54.5% | 54.8% | 24.3% | 45.7% |
| 32768 | **82.2%** | **83.4%** | 33.1% | **85.0%** |

Sparse, MDH, and cuSPARSE all climb to 82-85% achieved occupancy at
N=32768 - healthy GPU utilization once the grid is big enough to keep many
SMs busy. This makes sense: their launch configurations create more thread
blocks as N grows, and with 16 SMs to fill, more blocks means better
overlap.

## PPCG's occupancy number is real but misleading if read in isolation - and it's the clearest evidence yet of its 3D limitation

PPCG's achieved occupancy also climbs with N (7.9% -> 33.1%), which at a
glance looks like the same healthy scaling as the other three. **It isn't.**
PPCG's 3D launch config is `block(4, 4, M)`, `grid(1, 1)` at every size
tested (established in the roofline pass) - literally one block, always.
Its block *size* grows with M (since `block.z = M`), so a single block
occupies more of one SM's warp slots as M grows - but it is still, at
every size, exactly **one block on one SM**, with the GPU's other 15 SMs
sitting completely idle:

| M | threads/block | warps/block | predicted occupancy (1 block, 1 SM) | measured |
|---|---:|---:|---:|---:|
| 8 | 128 | 4.0 | 8.3% | 7.9% |
| 16 | 256 | 8.0 | 16.7% | 16.0% |
| 24 | 384 | 12.0 | 25.0% | 24.3% |
| 32 | 512 | 16.0 | 33.3% | 33.1% |

The predicted-from-block-size numbers match the measured occupancy almost
exactly at every size - direct, mechanical confirmation that PPCG's
"occupancy" is entirely an artifact of one block's warp count, not GPU-wide
utilization. **PPCG uses at most 1 of this GPU's 16 SMs (6.25% of the
device) at every 3D size tested** - this is a stronger,
more concrete statement than the roofline pass's low bandwidth/compute
percentages could make on their own, since those numbers alone could in
principle be explained by "PPCG's kernel body just does less work per
byte." The occupancy data rules that out: it's not that PPCG does less
work per SM, it's that PPCG almost never uses more than one SM.

## What this means

This strengthens the existing roofline/multigrid story rather than
standing alone: three
independent measurements now agree on the same conclusion about PPCG's 3D
generated schedule - `cudaEvent` timing (slower), roofline bandwidth/compute
(near-zero utilization), and now occupancy (literally one SM used,
mechanically confirmed via block-size arithmetic). This is PPCG's
unmodified default schedule (no `--sizes` flag used anywhere in this
project, matching the 2D pipeline), not a tuning failure on our part - but
it describes PPCG's default schedule for this problem shape in 3D. With tile and block sizes tuned through `--sizes` (see the top-level README), PPCG launches several blocks.
