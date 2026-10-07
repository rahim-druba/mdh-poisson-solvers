# Real-world application: heat conduction with a localized source

The Poisson equation is usually introduced as modeling the steady-state
temperature distribution on a plate. Every solver in this project so far
has only used that framing as flavor text before switching to a synthetic
verification problem chosen for having a known closed-form answer. This
demonstrates the actual physical scenario instead.

## The scenario

Steady-state heat conduction on a unit-square plate, same PDE
(`-Δu = f`) and grid (64x64 interior, N=4096) as
every solver in this project, but with physically-motivated inputs instead
of the synthetic verification ones:

- **Source**: `f(x,y) = 100 * exp(-((x-0.5)^2+(y-0.5)^2)/(2*0.1^2))` - a
  Gaussian heating element at the plate's center. Unlike the verification
  problem's constant `f=-4`, this is spatially varying - the first time
  in this project the source term `f` is used non-trivially rather than
  as a constant chosen to make a polynomial's algebra work out.
- **Boundary**: `g=0` on all four edges - the frame held at a fixed
  reference temperature (e.g. conduction to a heat sink).

## No closed-form solution - cross-method verification instead

A Gaussian source has no closed-form Poisson solution on a bounded square
(normal - that's why numerical methods exist for real problems). Solved
independently with two solvers already built and verified elsewhere in
this project - hand-written CSR (`kernel_sparse_heat.cu`) and MDH
matrix-free (`kernel_mdh_heat.cu`, reusing the exact same compiled
`cg_matvec_1.cu` kernel used everywhere else at N=4096) - and checked for
agreement, the standard approach when no analytical answer exists.

## Results

| | CSR | MDH |
|---|---:|---:|
| Iterations to converge | 99 | 99 |
| Peak temperature | 1.6257 | 1.6257 |
| Peak location (grid) | (31,31) | (31,31) |
| Min temperature | 0.0016 | 0.0016 |

**Cross-verification: max abs difference between the two independently
computed solutions = 2.0e-06** (mean 2.1e-07) - the same float32
rounding-level agreement every other cross-method check in this project
shows. Both solvers land on the identical iteration count, identical peak
temperature, identical peak location. This confirms that MDH
solves a physical problem correctly, not just the polynomial
chosen for easy verification.

**Physical sanity**: peak temperature at grid (31,31) -> physical
(0.492, 0.508), close to the source center (0.5, 0.5); minimum
temperature (0.0016) near the imposed `g=0` boundary; smooth radial
falloff visible in the rendered field.

## The result

![heatmap](heatmap.png)

A clean, smooth, radially-symmetric hot spot centered on the plate,
fading to the cold boundary - exactly the expected physics for a localized
heat source with a cooled frame.

## Why this matters

The same MDH-generated kernel already verified against a synthetic
polynomial is shown here solving a physical scenario (heat
conduction with a localized source), cross-checked against an independent
hand-written solver rather than a known formula, with a rendered result a
reader can look at rather than only an error table. No kernel code changed
to produce this - only the RHS construction differs from the verification
solvers, which is the point: the matrix-free MDH kernel is solving the
actual PDE operator, not something tuned to the verification problem
specifically.
