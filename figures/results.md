# Figures

Trends across problem size are much easier to read as a plot than as a
table, so these four figures cover the size-sweep results from across this
project that were only ever presented as tables. No new experiments; every
number here is transcribed from the `results.md` it's sourced from.

| Figure | Shows | Source data |
|---|---|---|
| `fig1_cg_iterations_vs_n.png` | CG iteration count grows with N, in both 2D and 3D | `../tables/full_cg/results.md`, `../5_3d_extension/results.md` |
| `fig2_multigrid_vs_cg.png` | Multigrid vs best-CG solve time - the gap widens from 3.3x to 8.5x as N grows | `../scaling_analysis/results.md`, `../6_multigrid/results.md` |
| `fig3_weak_scaling_efficiency.png` | Time-per-unknown: CG improves then reverses at large N; multigrid falls monotonically (24x, N=961->65025) | `../scaling_analysis/results.md` |
| `fig4_roofline_bandwidth_vs_n.png` | % of peak memory bandwidth vs N - CSR methods climb toward saturation (73%/57%); MDH stays low, goes compute-bound instead | `../roofline/results_3d.md` |

Regenerate with `python3 plot_all.py` (needs matplotlib - available in the
`torch310` conda env on this machine, confirmed present, not a new
dependency).
