#!/usr/bin/env python3
"""Cross-verify the CSR and MDH heat-conduction solutions (no closed-form
answer exists for a Gaussian source, so independent-method agreement is
the correctness check -- see results.md), then render a heatmap."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_grid(path):
    with open(path) as f:
        rows, cols = map(int, f.readline().split())
        data = np.loadtxt(f)
    assert data.shape == (rows, cols), f"{path}: expected {rows}x{cols}, got {data.shape}"
    return data

sparse = load_grid("solution_sparse.txt")
mdh = load_grid("solution_mdh.txt")

diff = np.abs(sparse - mdh)
max_diff = diff.max()
print(f"Grid shape: {sparse.shape}")
print(f"Max abs difference between CSR and MDH solutions: {max_diff:.6e}")
print(f"Mean abs difference: {diff.mean():.6e}")
print(f"CSR peak temp: {sparse.max():.4f} at {np.unravel_index(sparse.argmax(), sparse.shape)}")
print(f"MDH peak temp: {mdh.max():.4f} at {np.unravel_index(mdh.argmax(), mdh.shape)}")
print(f"CSR min temp: {sparse.min():.4f}, MDH min temp: {mdh.min():.4f}")

TOL = 1e-4
status = "PASS" if max_diff < TOL else "FAIL"
print(f"\nCross-verification (tol={TOL:.0e}): {status}")

# Heatmap (average of the two -- they agree to within float32 rounding, so
# either alone would look identical)
field = (sparse + mdh) / 2.0
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(field, cmap="inferno", origin="lower", extent=[0, 1, 0, 1])
ax.set_title("Steady-state heat conduction, Gaussian source\n(2D Poisson solve, MDH matrix-free kernel)")
ax.set_xlabel("x")
ax.set_ylabel("y")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Temperature (arbitrary units)")
fig.tight_layout()
fig.savefig("heatmap.png", dpi=150)
print("Wrote heatmap.png")
