# Hand-tuned matrix-free baseline (reviewer issue #3)

Same operator, representation (unpadded flat fp32 vector, matrix-free), CPU
oracle and timing as the MDH kernel. RTX 3050 Laptop (sm_86, 16 SMs, 6001 MHz
memory, 128-bit bus -> 192 GB/s theoretical peak; nvidia-smi: 4096 MiB), CUDA
11.7, driver 535.230.02, **persistence mode enabled**.

Protocol: >=300 ms warmup, then 10 repetitions x 200 timed launches (cudaEvent),
first repetition dropped, 9 averaged (the paper's protocol). Relative standard
deviation of the hand-tuned numbers: 0.00-0.32%. Every kernel and config is
checked against the CPU reference (max abs error <= 1.2e-5; no config marked WRONG).

Reproduce: `./run_all.sh` (or `mdh_tune_sweep.sh`, `build_and_run.sh` per size).
Raw output: `raw/`; every config's timing is in `raw/<dim>_<side>.txt` (hand-written)
and `raw/mdh_tune_<dim>_<side>.txt` (MDH search).

## Equal tuning effort
- **Hand-written**: ~30 configs/size over three families (naive; register-blocked
  `__ldg` sliding window; shared-memory tile+halo, both marching along the slow axis)
  with block shape and rows/planes per thread swept. Best correct config reported.
- **MDH**: the generated kernel's own tuning knobs swept per size: tile size L_CB per
  dimension, per-thread coarsening P_CB on the slowest dim, L1/P cache on/off
  (60 configs in 2D, 112-176 in 3D). All configs ranked with 3 reps, top 3 re-timed with
  the full protocol. "MDH untuned" = the fixed config used by the earlier tables
  (16x16 in 2D, 8x8x8 in 3D).

## 2D 5-point (N = side^2)

| side | N | naive (ms) | hand-tuned (ms) | best config | MDH untuned (ms) | MDH tuned (ms) | tuned config (L_CB, P_CB, cache) | untuned / HT | tuned / HT |
|---:|---:|---:|---:|---|---:|---:|---|---:|---:|
| 64 | 4,096 | 0.00139 | 0.00139 | reg 32x8 cy1 | 0.00162 | 0.00145 | 8x32, 2, off | 1.17 | 1.04 |
| 512 | 262,144 | 0.01370 | 0.01188 | reg 256x1 cy4 | 0.01455 | 0.01137 | 4x64, 2, off | 1.22 | 0.96 |
| 2048 | 4,194,304 | 0.19254 | 0.18494 | reg 128x2 cy2 | 0.20729 | 0.18484 | 8x64, 2, off | 1.12 | 1.00 |
| 4096 | 16,777,216 | 0.76373 | 0.73334 | reg 32x4 cy2 | 0.81954 | 0.73353 | 4x64, 2, off | 1.12 | 1.00 |

## 3D 7-point (N = side^3)

| side | N | naive (ms) | hand-tuned (ms) | best config | MDH untuned (ms) | MDH tuned (ms) | tuned config (L_CB, P_CB, cache) | untuned / HT | tuned / HT |
|---:|---:|---:|---:|---|---:|---:|---|---:|---:|
| 16 | 4,096 | 0.00151 | 0.00156 | reg 32x8 ci1 | 0.00190 | 0.00147 | 2x8x16, 2, off | 1.22 | 0.94 |
| 32 | 32,768 | 0.00229 | 0.00214 | reg 32x8 ci4 | 0.00334 | 0.00231 | 8x4x16, 2, off | 1.56 | 1.08 |
| 64 | 262,144 | 0.01394 | 0.01223 | reg 32x4 ci4 | 0.01986 | 0.01308 | 2x4x32, 2, off | 1.62 | 1.07 |
| 128 | 2,097,152 | 0.09755 | 0.09428 | reg 32x4 ci4 | 0.15427 | 0.09348 | 2x4x32, 2, off | 1.64 | 0.99 |
| 256 | 16,777,216 | 0.88798 | 0.78041 | reg 32x4 ci16 | 1.31257 | 0.77993 | 16x2x32, 2, off | 1.68 | 1.00 |

Hand-tuned bandwidth at the largest sizes (8N bytes / time): 183 GB/s in 2D
(side 4096), 172 GB/s in 3D (side 256), i.e. 90-95% of the 192 GB/s theoretical peak.
Sizes up to 64^2 / 16^3 are launch-overhead-bound (~1.4-1.6 us), so ratios there
are within launch noise.

## What this shows
1. **Once tuned with comparable effort, MDH's generated code matches the expert
   kernel**: 0.94-1.08x at every size, and within 1% at the large sizes
   (2D >= 2048, 3D >= 128) where the kernel is bandwidth-bound.
2. **The earlier 3D gap (1.6-1.7x) was almost entirely the untuned config**, not the
   generated code. 2D: untuned MDH was 12-22% off.
3. A naive one-thread-per-point kernel is 4-14% behind the ceiling at the largest 2D/3D
   sizes (3D 256: 0.888 vs 0.780 ms).

## Caveats to state in the paper
- The tuned MDH configs come from *this* sweep over MDH's macro space (all tuned
  configs use P_CB=2 coarsening and no cache). Report them (they are in `raw/`)
  and do not call them the output of the paper's auto-tuner unless they are.
- **The paper's current MDH tables use the untuned configs** (e.g. 8x8x8 in 3D per
  `5_3d_extension/build_and_run_sweep_3d.sh`). Check whether the 3D results
  (Tables 6/7, the "8x over PPCG" claim, the roofline/occupancy sections) were
  produced with an untuned config. If so they understate MDH by up to 1.7x and the
  "compute-bound at 40%" explanation needs re-measuring with the tuned kernel
  (the hand-written kernel is bandwidth-bound).
- fp32, unpadded layout, one GPU (RTX 3050). The same scripts (`ARCH=sm_120`) run
  on the 5080/5090.
- Paper's roofline "peak 150-160 GB/s" is below the 183 GB/s measured here;
  recheck the percent-of-peak figures.
