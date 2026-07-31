#!/bin/bash
# GPU occupancy profiling for the multigrid solver -- closes the gap
# flagged in ../scaling_analysis/results.md ("multigrid's own GPU-
# utilization side isn't measured yet"). Profiles jacobi_smooth_kernel at
# the FINEST level of the V-cycle (launch index 1, the 2nd pre-smoothing
# sweep -- deterministically at level 0 regardless of how many total
# levels a given size has, since the down-sweep always starts at the
# finest level; skips launch 0 to avoid any single cold-start blip, same
# reasoning as the CG occupancy pass's launch-skip=5).
#
# Needs root -- run with sudo:
#   sudo bash profile_occupancy_mg.sh
set -e
cd "$(dirname "$0")"

NCU=/usr/local/cuda-11.7/bin/ncu
EXT=../6_multigrid

for M in 31 63 127 255; do
  N=$((M * M))
  echo "=== M=$M (N=$N) ==="
  $NCU --target-processes all --launch-skip 1 --launch-count 1 \
    --kernel-name jacobi_smooth_kernel --section Occupancy \
    "$EXT/mg_$M" > "occ_mg_${M}.txt" 2>&1
  tail -20 "occ_mg_${M}.txt"
  echo
done

echo "All done. Reports: occ_mg_{31,63,127,255}.txt"
