# Cross-architecture check: hand-tuned baseline vs MDH on RTX 5090

Same harness, protocol and kernels as `results.md` (RTX 3050 Laptop), built with `-arch=sm_120`.
Machine: titan-MS-7E48, 2x RTX 5090 (GPU 1 used: 32 GB, 170 SMs, 512-bit GDDR7, ~1792 GB/s), driver 595.84,
CUDA 12.9.86 (conda-forge, private install), GCC 13.3.0, Ubuntu 24.04.4, persistence mode on, no other GPU processes
recorded after any size. Under load the clock was median 2820 MHz (min 1057 MHz when the board reached its ~600 W limit
in bandwidth-heavy phases), temperature at most 61 C. Protocol: >=300 ms warm-up, 10 x 200 launches, first dropped, 9 averaged.
All kernels verified against the CPU reference (no WRONG line in any log). Raw logs: `raw_5090/`.
MDH "tuned" = best of the same parameter sweep as on the 3050 (tile sizes, coarsening, cache); "untuned" = fixed 16x16 (2D) / 8x8x8 (3D).

## 2D 5-point (N = side^2), ms

| side | N | naive | hand-tuned | best config | MDH untuned | MDH tuned | tuned config (L_CB, P, cache) | untuned/HT | tuned/HT |
|---:|---:|---:|---:|---|---:|---:|---|---:|---:|
| 512 | 262,144 | 0.00132 | 0.00130 | reg 128x2, 1 row | 0.00142 | 0.00131 | 16x32, 2, off | 1.09 | 1.01 |
| 2048 | 4,194,304 | 0.00859 | 0.00680 | reg 128x2, 4 rows | 0.01016 | 0.00744 | 8x64, 2, off | 1.49 | 1.09 |
| 4096 | 16,777,216 | 0.08186 | 0.07776 | reg 256x1, 8 rows | 0.09674 | 0.07791 | 8x64, 2, off | 1.24 | 1.00 |
| 8192 | 67,108,864 | 0.35049 | 0.35045 | reg 64x4, 1 row | 0.46631 | 0.34178 | 8x64, 1, off | 1.33 | 0.98 |
| 16384 | 268,435,456 | 1.40427 | 1.40304 | reg 128x2, 1 row | 1.86080 | 1.36488 | 8x64, 1, off | 1.33 | 0.97 |

## 3D 7-point (N = side^3), ms

| side | N | naive | hand-tuned | best config | MDH untuned | MDH tuned | tuned config (L_CB, P, cache) | untuned/HT | tuned/HT |
|---:|---:|---:|---:|---|---:|---:|---|---:|---:|
| 64 | 262,144 | 0.00137 | 0.00144 | reg 32x8, 1 plane | 0.00194 | 0.00131 | 2x4x64, 2, off | 1.35 | 0.91 |
| 128 | 2,097,152 | 0.00483 | 0.00433 | reg 64x4, 4 planes | 0.00997 | 0.00474 | 2x4x64, 2, off | 2.30 | 1.09 |
| 256 | 16,777,216 | 0.08276 | 0.07902 | reg 64x4, 8 planes | 0.14535 | 0.07924 | 2x8x64, 2, off | 1.84 | 1.00 |
| 512 | 134,217,728 | 0.88431 | 0.70469 | smem 32x4, 8 planes | 1.18530 | 0.70325 | 2x2x64, 2, off | 1.68 | 1.00 |

## What this shows
1. Tuned MDH again matches the hand-tuned kernel: 0.91-1.09 times its run time at every size, 0.97-1.00 in the large
   bandwidth-bound cases (2D side >= 4096, 3D side >= 256).
2. Untuned MDH is further from the ceiling here (up to 1.49x in 2D, 2.30x in 3D) than on the 3050 (1.12-1.22x / 1.56-1.68x),
   so tuning matters more on the newer GPU.
3. The best parameters differ between GPUs (5090: 8x64 tiles in 2D, 2x4x64..2x8x64 in 3D; 3050: 4x64..8x32 in 2D, 2x4x32..16x2x32 in 3D).
4. At the large sizes the hand-tuned kernels reach 85-96% of the 1792 GB/s peak (2D 4096^2: 96%, 2D 16384^2: 85%, 3D 256^3: 95%, 3D 512^3: 85%).
   The GB/s of 2D side 2048 and 3D side 128 (tens of MB) exceed the DRAM peak because they are served from the 96 MB L2 cache, not DRAM.

## Caveats
- One run per size (variance within a run is reported in the raw logs); sizes up to 512^2 / 64^3 are launch-overhead-bound.
- The tuned MDH configuration was selected on the same machine and harness; MDH was tuned by a macro sweep, not a full ATF run.
- The 2D 64x64 smoke test has no tuned entry (not tuned).
- Compiler differs from the 3050 runs (CUDA 12.9 / GCC 13.3 here vs CUDA 11.7 / GCC 9.4): the cross-GPU comparison also changes the toolchain.
- CSR, cuSPARSE and PPCG were not run on the 5090 yet; only hand-tuned vs MDH.
