#!/bin/bash
# GPU occupancy profiling pass. Uses Nsight Compute's built-in Occupancy section (achieved
# occupancy, theoretical occupancy, and the limiting factor -- registers,
# shared memory, block size, or warps/SM) rather than hand-picked metric
# names, since metric names can't be verified without perf-counter access
# (see the roofline pass's notes) and sections are the more stable API
# across ncu versions.
#
# Profiles the same steady-state kernel launches as the roofline passes:
# all four 2D matvec kernels at N=4096, plus all four methods across the
# full 3D sweep (N=512/4096/13824/32768) -- reuses the exact same
# already-verified binaries, no new code.
#
# Needs root -- run with sudo:
#   sudo bash profile_occupancy.sh
set -e
cd "$(dirname "$0")"

NCU=/usr/local/cuda-11.7/bin/ncu
EXT=../5_3d_extension

profile() {
  local label=$1 kernel_filter=$2 binary=$3 outfile=$4
  echo "--- $label ---"
  if [ -z "$kernel_filter" ]; then
    $NCU --target-processes all --launch-skip 5 --launch-count 1 \
      --section Occupancy "$binary" > "$outfile" 2>&1
  else
    $NCU --target-processes all --launch-skip 5 --launch-count 1 \
      --kernel-name "$kernel_filter" --section Occupancy "$binary" > "$outfile" 2>&1
  fi
  tail -20 "$outfile"
}

echo "=== 2D, N=4096 ==="
profile "sparse"   "spmv_csr_kernel" "../1_sparse_rewrite/kernel_sparse"   "occ_sparse_2d.txt"
profile "mdh"      "cg_matvec_1"     "../2_mdh_sparse/kernel_mdh_sparse"   "occ_mdh_2d.txt"
profile "ppcg"     "kernel0"         "../3_ppcg/kernel_ppcg_sparse"        "occ_ppcg_2d.txt"
profile "cusparse" ""                "../4_cusparse/kernel_cusparse"       "occ_cusparse_2d.txt"

for M in 8 16 24 32; do
  N=$((M * M * M))
  echo
  echo "=== 3D, M=$M (N=$N) ==="
  profile "sparse"   "spmv_csr_kernel" "$EXT/sparse_$M"   "occ_sparse_3d_${M}.txt"
  profile "mdh"      "cg_matvec_3d_1"  "$EXT/mdh_$M"      "occ_mdh_3d_${M}.txt"
  profile "ppcg"     "kernel0"         "$EXT/ppcg_$M"     "occ_ppcg_3d_${M}.txt"
  profile "cusparse" ""                "$EXT/cusparse_$M" "occ_cusparse_3d_${M}.txt"
done

echo
echo "All done. Reports: occ_*.txt in this directory."
