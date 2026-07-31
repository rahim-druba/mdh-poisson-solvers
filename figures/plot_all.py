#!/usr/bin/env python3
"""Speedup and scaling figures from data already measured elsewhere in
this project. No new experiments; every number here is transcribed from
the corresponding results.md."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# ---------------------------------------------------------------------
# Figure 1: CG iteration count vs N (2D and 3D) -- source: ../tables/full_cg/results.md, ../5_3d_extension/results.md
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
n2d = [512, 1024, 2048, 4096]
it2d = [75, 86, 144, 192]
n3d = [512, 4096, 13824, 32768]
it3d = [27, 58, 90, 131]  # midpoints of the small per-method spread reported

ax.plot(n2d, it2d, "o-", label="2D CG", color="#d62728")
ax.plot(n3d, it3d, "s-", label="3D CG", color="#1f77b4")
ax.set_xscale("log")
ax.set_xlabel("N (unknowns)")
ax.set_ylabel("iterations to converge")
ax.set_title("CG iteration count grows with problem size\n(condition number scales with N)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig1_cg_iterations_vs_n.png")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: multigrid vs best-CG time vs N -- source: ../scaling_analysis/results.md, ../6_multigrid/results.md
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
n_cg2d = [512, 1024, 2048, 4096]
t_cg2d_best = [1.1617, 1.5082, 2.9023, 4.9836]  # fastest CG variant at each size
n_mg = [961, 3969, 16129, 65025]
t_mg = [0.4570, 0.5839, 0.8281, 1.2780]

ax.plot(n_cg2d, t_cg2d_best, "o-", label="best CG variant (2D)", color="#d62728")
ax.plot(n_mg, t_mg, "^-", label="multigrid (2D)", color="#2ca02c")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("N (unknowns)")
ax.set_ylabel("solve time (ms)")
ax.set_title("Multigrid vs CG: the gap widens with N\n(3.3x faster at N~1024, 8.5x faster at N~4096)")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig2_multigrid_vs_cg.png")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 3: weak-scaling efficiency (time/N) -- source: ../scaling_analysis/results.md
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
n_cg2d_w = [512, 1024, 2048, 4096]
eff_cg2d = [2.436, 1.494, 1.417, 1.217]  # sparse, us/unknown
n_cg3d_w = [512, 4096, 13824, 32768]
eff_cg3d = [0.903, 0.358, 0.338, 0.404]  # sparse, us/unknown -- note the reversal at 32768
n_mg_w = [961, 3969, 16129, 65025]
eff_mg = [0.4755, 0.1471, 0.0513, 0.0197]

ax.plot(n_cg2d_w, eff_cg2d, "o-", label="2D CG (sparse)", color="#d62728")
ax.plot(n_cg3d_w, eff_cg3d, "s-", label="3D CG (sparse)", color="#ff7f0e")
ax.plot(n_mg_w, eff_mg, "^-", label="multigrid", color="#2ca02c")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("N (unknowns)")
ax.set_ylabel("time per unknown (us)")
ax.set_title("Weak scaling: CG improves then reverses;\nmultigrid falls monotonically (24x, 961->65025)")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig3_weak_scaling_efficiency.png")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 4: roofline, % of peak bandwidth vs N -- source: ../roofline/results_3d.md
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
n_rl = [512, 4096, 13824, 32768]
sparse_bw = [4.2, 28.3, 56.3, 72.9]
cusparse_bw = [2.9, 21.5, 47.0, 57.4]
mdh_bw = [0.5, 3.4, 9.6, 14.2]
ppcg_bw = [0.7, 1.5, 2.2, 2.3]

ax.plot(n_rl, sparse_bw, "o-", label="sparse (CSR)", color="#d62728")
ax.plot(n_rl, cusparse_bw, "d-", label="cusparse (CSR)", color="#9467bd")
ax.plot(n_rl, mdh_bw, "s-", label="mdh (matrix-free)", color="#1f77b4")
ax.plot(n_rl, ppcg_bw, "v-", label="ppcg (matrix-free)", color="#8c564b")
ax.set_xscale("log")
ax.set_xlabel("N (unknowns, 3D sweep)")
ax.set_ylabel("% of peak memory bandwidth")
ax.set_title("CSR becomes genuinely memory-bound at scale;\nMDH stays low (goes compute-bound instead)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig4_roofline_bandwidth_vs_n.png")
plt.close(fig)

print("Wrote fig1_cg_iterations_vs_n.png, fig2_multigrid_vs_cg.png, "
      "fig3_weak_scaling_efficiency.png, fig4_roofline_bandwidth_vs_n.png")
