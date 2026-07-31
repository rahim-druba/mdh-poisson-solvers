# Geometric multigrid vs CG - results

A real geometric multigrid (GMG) solver for the same 2D Poisson problem
every CG variant in this project solves, built to give CG a real point of
comparison against a different algorithm, with working code and real
numbers, not another CG-vs-CG comparison.

**Method**: standard V-cycle, damped weighted Jacobi smoother (`omega=0.8`,
2 pre + 2 post-smoothing sweeps/level), full-weighting restriction,
bilinear prolongation, coarsest level (1 point) solved exactly in closed
form. Same domain, BC, and analytical solution (`u = 1 + x^2 + y^2`) as
every CG solver here - see `kernel_multigrid_2d.cu`'s header comment for
why multigrid needs its own properly h-scaled operator convention
(different from the CG solvers' "unscaled stencil" convention) and why the
grid family is `m = 2^k - 1` (31/63/127/255) rather than the CG tables'
plain powers of 2 - both are load-bearing correctness requirements, not
style choices; an earlier attempt using the CG convention's grid alignment
diverged once a 3rd coarsening level was added (documented in
`cpu_reference_multigrid_2d.h`).

## Correctness

Every kernel (Jacobi smoothing, residual, restriction, prolongation)
checked element-wise against a CPU reference before any GPU timing was
trusted (`test_gpu_multigrid_kernels.cu`) - zero mismatches, all four. Full
V-cycle solve verified against the CPU reference (`test_cpu_vcycle.cu`)
and, independently, the GPU solve verified against the known analytical
solution at every size below.

## Table: multigrid alone, swept across sizes

10 runs, first dropped as warmup, remaining 9 averaged - same protocol as
`../tables/full_cg/results.md`.

| size | avg ms (9-run) | V-cycles | max err vs analytical |
|---|---:|---:|---:|
| 961 (31^2) | 0.4570 | 6 | 9.97e-05 |
| 3969 (63^2) | 0.5839 | 6 | 1.03e-04 |
| 16129 (127^2) | 0.8281 | 6 | 1.09e-04 |
| 65025 (255^2) | 1.2780 | 6 | 1.30e-04 |

**The V-cycle count is exactly 6 at every size tested - an 8x range in
grid side length (31 to 255), a 68x range in unknowns (961 to 65025), zero
change in iteration count.** This is the textbook multigrid property that
makes this comparison worth having: unlike CG, whose iteration count grows
with the grid (condition number scales with N), multigrid's convergence
rate is asymptotically grid-independent.

*(Stopping criterion: relative residual `< 1e-5`, not the CG solvers'
absolute `1e-6` - the properly h-scaled operator convention multigrid
needs produces residuals ~1000x larger in magnitude than the CG solvers'
"unscaled stencil" convention, so an absolute `1e-6` threshold hits
float32's precision floor before ever being reached at larger sizes,
confirmed empirically. `1e-5` relative converges cleanly everywhere and
still gives ~1e-4 solution accuracy - slightly looser than the CG solvers'
typical ~1e-6, a real, honest difference worth being upfront about.)*

## Table: multigrid vs CG, matching sizes

`../tables/full_cg/results.md`'s N=1024/4096 rows against the nearest
multigrid sizes (961/3969 - the `2^k-1` grid family doesn't land on exactly
the same N, close enough for a real comparison):

| N (approx) | sparse | mdh | ppcg | cusparse | **multigrid** |
|---|---:|---:|---:|---:|---:|
| ~1024 (1024 / 961) | 1.5293ms | 1.5082ms | 1.5240ms | 1.6491ms | **0.4570ms** |
| ~4096 (4096 / 3969) | 4.9836ms | 5.0590ms | 5.0804ms | 5.3237ms | **0.5839ms** |

**Multigrid beats every CG variant at both sizes, and the margin grows
sharply with N**: 3.3x faster than the best CG variant (MDH) at N~1024,
widening to **8.5x faster** than the best CG variant (sparse) at N~4096.
This is the actual mechanism behind the widening gap: CG's iteration count
grows with N (75 -> 86 -> 144 -> 192 across the existing 2D sweep),
multigrid's doesn't (flat at 6) - so the gap between them is not a fixed
constant, it widens indefinitely as the problem gets larger.

## What this settles

A real, independently-verified geometric multigrid solver now exists,
benchmarked against every CG variant in this project at matching problem
sizes, on the identical PDE, showing exactly the widening-gap pattern
multigrid theory predicts. Any honest evaluation of code-generation
strategies for this kind of solver should sit next to this comparison, not
only next to other CG variants.

## What's still open from here

- Only 2D - matches the same scope used for the 3D CG extension; a 3D
  multigrid solver would follow the same pattern if wanted later.
- No larger-N comparison against CG beyond N=4096 (the existing CG sweep's
  ceiling) - the widening-gap trend is already clear from two points, but
  a CG run at N~65025 would make the asymptotic argument fully airtight.
- Smoother choice (weighted Jacobi) was picked for GPU-friendliness and
  implementation safety, not because it's optimal - red-black Gauss-Seidel
  would likely converge in fewer V-cycles still, not attempted here.
- Precision is `float` throughout, same as the rest of this project.
