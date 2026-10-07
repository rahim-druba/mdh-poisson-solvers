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
t_cg2d_best = [0.9009, 1.2512, 2.9202, 5.1122]  # fastest CG variant at each size
n_mg = [961, 3969, 16129, 65025]
t_mg = [0.443, 0.5868, 0.8534, 1.2638]

ax.plot(n_cg2d, t_cg2d_best, "o-", label="best CG variant (2D)", color="#d62728")
ax.plot(n_mg, t_mg, "^-", label="multigrid (2D)", color="#2ca02c")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("N (unknowns)")
ax.set_ylabel("solve time (ms)")
ax.set_title("Multigrid vs CG: the gap widens with N\n(2.8x faster at N~1024, 8.7x faster at N~4096)")
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
eff_cg2d = [2.529, 1.523, 1.479, 1.271]  # sparse, us/unknown
n_cg3d_w = [512, 4096, 13824, 32768]
eff_cg3d = [0.912, 0.369, 0.354, 0.4]  # sparse, us/unknown -- note the reversal at 32768
n_mg_w = [961, 3969, 16129, 65025]
eff_mg = [0.461, 0.1478, 0.0529, 0.0194]

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
# Figure 4: DRAM bandwidth reached vs N (3D), in percent of the 192 GB/s theoretical peak
# source: ../roofline/results_bandwidth_3d.md (raw Nsight Compute outputs)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
n_rl = [512, 4096, 13824, 32768]
series = [
    ("CSR (hand-written)", [3.4, 25.3, 50.2, 59.1], "o-", "#d62728"),
    ("cuSPARSE", [2.3, 18.4, 40.2, 51.3], "d-", "#9467bd"),
    ("MDH, untuned", [0.4, 2.7, 7.9, 12.6], "s--", "#8fb8de"),
    ("MDH, tuned", [0.4, 3.0, 8.6, 17.9], "s-", "#1f77b4"),
    ("hand-tuned matrix-free", [0.4, 2.9, 9.1, 15.0], "^-", "#2ca02c"),
    ("PPCG, default schedule", [0.5, 1.4, 2.1, 2.2], "v-", "#8c564b"),
]
for label, y, style, color in series:
    ax.plot(n_rl, y, style, label=label, color=color)
ax.set_xscale("log")
ax.set_xlabel("N (unknowns, 3D sweep)")
ax.set_ylabel("DRAM bandwidth, % of 192 GB/s peak")
ax.set_title("RTX 3050: CSR approaches the bandwidth limit,\nmatrix-free kernels stay low at these sizes")
ax.legend(fontsize=9.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig4_roofline_bandwidth_vs_n.png")
plt.close(fig)

print("Wrote fig1_cg_iterations_vs_n.png, fig2_multigrid_vs_cg.png, "
      "fig3_weak_scaling_efficiency.png, fig4_roofline_bandwidth_vs_n.png")
