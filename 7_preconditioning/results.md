# Preconditioned CG - results

Jacobi and ILU(0) preconditioning, tested against plain CG to see whether
either speeds up convergence for this operator. Both are
implemented, verified, and swept across four 2D grid sizes - same domain, BC, and
analytical solution (`u = 1 + x^2 + y^2`) as every solver in this project,
CSR baseline (`build_poisson_csr`, identical to `1_sparse_rewrite/kernel_sparse.cu`).

## Correctness

Both variants converge to the analytical solution within the same
float32-rounding-level error every other solver in this project shows
(1.2e-6 to 5.0e-6), at every size tested. ILU(0) factorization reports no
zero pivots at any size.

## Table: iterations and wall-clock time, all three methods, swept

10 runs, first dropped as warmup, remaining 9 averaged - same protocol as
`../tables/full_cg/results.md`. Square grids (16x16 to 128x128, N=256 to
16384) to keep the FULLN convention identical to `1_sparse_rewrite/kernel_sparse.cu`.

| N | plain CG (ms) | Jacobi PCG (ms) | ILU(0) PCG (ms) | plain iters | Jacobi iters | ILU(0) iters |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 0.7833 | 0.7884 | 1.5307 | 43 | **43** | 17 |
| 1024 | 1.6656 | 1.7009 | 4.7641 | 86 | **86** | 30 |
| 4096 | 5.3678 | 5.5313 | 22.3424 | 192 | **192** | 56 |
| 16384 | 26.5364 | 29.8967 | 121.8384 | 406 | **406** | 109 |

## Finding 1: Jacobi preconditioning is confirmed inert - exactly, not approximately

**Jacobi's iteration count matches plain CG exactly at every single size**
(43/43, 86/86, 192/192, 406/406) - not close, identical. This confirms the
prediction made before writing any code: the discrete 5-point Poisson
operator has a constant diagonal (`diag(A) = 4I` everywhere), so a
diagonal preconditioner is just a uniform scalar, which is algebraically
inert for CG (the scaling cancels in the `alpha`/`beta` ratios). Jacobi
preconditioning has no effect on this specific problem - this is
mathematically expected for constant-coefficient problems, not a bug or a
weak implementation, and it's now backed by measurement rather than
assertion.

## Finding 2: ILU(0) reduces iterations substantially - and the benefit grows with N

| N | iteration reduction (plain/ILU0) |
|---|---:|
| 256 | 2.53x |
| 1024 | 2.87x |
| 4096 | 3.43x |
| 16384 | **3.72x** |

ILU(0) meaningfully improves convergence, and more so as the problem gets
larger - consistent with it countering the same condition-number growth
that makes plain CG's iteration count climb with N (documented in
`../scaling_analysis/results.md`).

## Finding 3: ILU(0) is a net wall-clock loss here - and gets worse with N, not better

| N | time slowdown (ILU0/plain) |
|---|---:|
| 256 | 1.95x |
| 1024 | 2.86x |
| 4096 | 4.16x |
| 16384 | **4.59x** |

Despite doing meaningfully fewer iterations, ILU(0) is **1.95x to 4.6x
slower in wall-clock time** than plain CG at every size tested, and the
slowdown *worsens* with N even as the iteration-count benefit improves -
the two trends move in opposite directions. Per-iteration cost tells the
story: at N=4096, ILU(0) averages ~0.40ms/iteration vs plain CG's
~0.028ms/iteration - about 15x more expensive per step. This is a known,
well-documented characteristic of GPU sparse triangular solves: unlike
`SpMV` (fully data-parallel), a triangular solve has
level-to-level data dependencies, making `cusparseSpSV` intrinsically far
less parallelizable on this hardware. The two extra triangular solves per
iteration (forward on `L`, backward on `U`) cost more than the iterations
saved, and that gap widens as the grid grows.

## Summary

Both preconditioners now have a real, verified answer instead of a guess:
- **Jacobi**: confirmed inert for this problem class, with real
  supporting data (exact iteration-count match at every size) - not
  something to silently skip.
- **ILU(0)**: works as a *convergence* accelerant (fewer iterations,
  reliably, with the benefit growing at scale) but is **not a wall-clock
  win on this GPU** for this problem size range, due to triangular-solve
  overhead outweighing the iteration savings. This reports both the iteration-count improvement and the
  wall-clock cost that comes with it.
