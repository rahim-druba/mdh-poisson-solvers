# Table 3: dense matrix multiplication (S = A*B), 4-way, size sweep

Square N x N x N GEMM, row-major layout (matching PPCG's/MDH's own
conventions). `cusparse` doesn't do dense GEMM, so **cuBLAS takes that slot**
here - matches what the original article's own Table 3 actually compared
against. `bench_matmul.cu`, one binary per size, `-DGEMM_N=<size>`.

Methodology: 2 warmup + 5 timed launches, `cudaEvent`, averaged - far fewer
than the matvec table's 200, because a single GEMM call here is itself
O(N^3) work (up to ~430ms/call at N=4096), not microseconds. Correctness:
50 spot-checked output elements against an O(N) CPU dot product each (a
full O(N^3) CPU reference would be impractically slow at N>=2048).

| size | naive (ms) | mdh (ms) | ppcg (ms) | cublas (ms) | correct (all 4) |
|---|---:|---:|---:|---:|:---:|
| 512  | 0.624   | 0.823   | 2.271   | **0.077**  | yes |
| 1024 | 4.928   | 6.466   | 41.973  | **0.442**  | yes |
| 2048 | 37.763  | 46.071  | 69.288  | **3.388**  | yes |
| 4096 | 283.820 | 369.788 | 435.918 | **29.299** | yes |

GFLOP/s at N=4096: naive 484, mdh 372, ppcg 315, **cublas 4691**.

## What this shows (and it's a genuinely different story than the matvec table)

- **cuBLAS wins by 10-15x at every size, no contest.** This is the expected
  outcome: cuBLAS is the right baseline for a dense GEMM specifically
  (unlike cuSPARSE, which is the right baseline for sparse SpMV instead).
  cuBLAS's SGEMM is a mature, extensively hand/auto-tuned vendor kernel
  (likely using tensor-core or highly specialized code paths this RTX 3050
  supports); no generated kernel here comes close.
- **Naive beats both MDH and PPCG at every size tested.** This is the
  opposite ordering from the matvec table, and worth taking seriously
  rather than glossing over: unlike the matvec case, both MDH's and PPCG's
  GEMM kernels here are running an **untuned first-guess tile
  configuration** (L_CB 64x64x16, reused as-is from prior int-typed work in
  `cg-ppcg-test`), not a searched/optimal one. Prior work in that same repo
  found a *hand-swept* MDH config that was 2.4x faster than PPCG on this
  exact kernel shape at N=2048 (609 vs 250 GOPS) - so both generated
  kernels here are likely leaving real performance on the table. The naive
  kernel's simplicity (no shared-memory synchronization overhead, and
  N=512-4096 working sets that partly fit this GPU's L2 cache) lets it
  win by default at this scale on this hardware, not because auto-tiling
  is a bad idea in general.
- **This table is the least connected to the CG rewrite itself** (as
  flagged before starting this) - CG never does a matmul - but it directly
  answers what "matrix multiplication" would show if reproduced properly
  against the correct baseline (cuBLAS, not cuSPARSE), and the honest
  result is that neither generator's default config beats a naive kernel
  here, let alone cuBLAS. A fair MDH/PPCG comparison on GEMM would need
  the tuning sweep this table skipped.
