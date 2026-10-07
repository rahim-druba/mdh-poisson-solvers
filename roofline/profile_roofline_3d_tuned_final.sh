#!/bin/bash
# Nsight Compute re-profile of the TUNED MDH kernel and the hand-tuned kernel at the 3D paper sizes (reviewer issue 2:
# is the tuned MDH kernel still "compute-bound"?). Uses the binaries of the final one-session full-solve run
# (handtuned_baseline/bin/cg_mdht_3d<M>, cg_ht0_3d<M>), built with -arch=sm_86 and the final tuned configuration.
# Needs GPU performance-counter access (root):   sudo bash /home/rahim/cg/roofline/profile_roofline_3d_tuned_final.sh
set -e
cd "$(dirname "$0")"
NCU=/usr/local/cuda-11.7/bin/ncu
BIN=../handtuned_baseline/bin
METRICS="gpu__time_duration.sum,dram__bytes.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum"
for M in 8 16 24 32; do
  echo "=== M=$M ==="
  $NCU --target-processes all --launch-skip 5 --launch-count 1 --kernel-name cg_matvec_3d_1 --metrics "$METRICS" \
    "$BIN/cg_mdht_3d$M" > "mdh_3d_${M}_tuned_ncu.txt" 2>&1
  $NCU --target-processes all --launch-skip 5 --launch-count 1 --kernel-name cg_matvec_3d_1 --section Occupancy \
    "$BIN/cg_mdht_3d$M" > "occ_mdh_3d_${M}_tuned.txt" 2>&1
  $NCU --target-processes all --launch-skip 5 --launch-count 1 --kernel-name regex:st_reg --metrics "$METRICS" \
    "$BIN/cg_ht0_3d$M" > "ht_3d_${M}_ncu.txt" 2>&1
  grep -E "dram__bytes.sum.per_second|dram__throughput|sm__throughput|gpu__time" "mdh_3d_${M}_tuned_ncu.txt" "ht_3d_${M}_ncu.txt"
done
echo "All done. Reports: mdh_3d_{8,16,24,32}_tuned_ncu.txt, occ_mdh_3d_*_tuned.txt, ht_3d_*_ncu.txt"
