#!/bin/bash
# Roofline re-profile at 3D sizes -- follow-up to profile_roofline.sh's 2D
# N=4096 finding that none of the kernels were near either roofline
# ceiling. Checks whether achieved bandwidth climbs at larger N, using the
# already-built, already-correctness-verified 3D solvers from
# ../5_3d_extension/ (sparse_M, mdh_M, ppcg_M, cusparse_M for M=8/16/24/32,
# i.e. N=512/4096/13824/32768).
#
# Needs GPU performance-counter access (root) -- run with sudo:
#   sudo bash profile_roofline_3d.sh
set -e
cd "$(dirname "$0")"

NCU=/usr/local/cuda-11.7/bin/ncu
METRICS="gpu__time_duration.sum,dram__bytes.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum"
EXT=../5_3d_extension

for M in 8 16 24 32; do
  N=$((M * M * M))
  echo "=== M=$M (N=$N) ==="

  echo "--- sparse ---"
  $NCU --target-processes all --launch-skip 5 --launch-count 1 \
    --kernel-name spmv_csr_kernel --metrics "$METRICS" \
    "$EXT/sparse_$M" > "sparse_3d_${M}_ncu.txt" 2>&1
  tail -12 "sparse_3d_${M}_ncu.txt"

  echo "--- mdh ---"
  $NCU --target-processes all --launch-skip 5 --launch-count 1 \
    --kernel-name cg_matvec_3d_1 --metrics "$METRICS" \
    "$EXT/mdh_$M" > "mdh_3d_${M}_ncu.txt" 2>&1
  tail -12 "mdh_3d_${M}_ncu.txt"

  echo "--- ppcg ---"
  $NCU --target-processes all --launch-skip 5 --launch-count 1 \
    --kernel-name kernel0 --metrics "$METRICS" \
    "$EXT/ppcg_$M" > "ppcg_3d_${M}_ncu.txt" 2>&1
  tail -12 "ppcg_3d_${M}_ncu.txt"

  echo "--- cusparse ---"
  $NCU --target-processes all --launch-skip 5 --launch-count 1 \
    --metrics "$METRICS" \
    "$EXT/cusparse_$M" > "cusparse_3d_${M}_ncu.txt" 2>&1
  tail -15 "cusparse_3d_${M}_ncu.txt"

  echo
done

echo "All done. Reports: {sparse,mdh,ppcg,cusparse}_3d_{8,16,24,32}_ncu.txt"
