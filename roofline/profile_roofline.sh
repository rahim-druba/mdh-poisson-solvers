#!/bin/bash
# Roofline profiling pass - measures real memory bandwidth utilization
# with Nsight Compute instead of relying on a theoretical estimate.
#
# Needs GPU performance-counter access, which needs root -- run this
# script with sudo, once, from your own terminal, from inside this folder:
#
#   sudo bash profile_roofline.sh
#
# Profiles one steady-state matvec kernel launch (skips the first few
# iterations to avoid any cold-start effects) from each of the four
# existing, already-correctness-verified 2D CG solvers at N=4096 --
# reuses the real binaries, no new/separate benchmark code. Output goes to
# one text file per method in this directory; nothing here modifies any
# of the profiled binaries or their source.
set -e
cd "$(dirname "$0")"

NCU=/usr/local/cuda-11.7/bin/ncu
METRICS="gpu__time_duration.sum,dram__bytes.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum"

echo "=== sparse (hand-written CSR) ==="
$NCU --target-processes all --launch-skip 5 --launch-count 1 \
  --kernel-name spmv_csr_kernel --metrics "$METRICS" \
  ../1_sparse_rewrite/kernel_sparse > sparse_ncu.txt 2>&1
tail -20 sparse_ncu.txt

echo
echo "=== mdh (matrix-free) ==="
$NCU --target-processes all --launch-skip 5 --launch-count 1 \
  --kernel-name cg_matvec_1 --metrics "$METRICS" \
  ../2_mdh_sparse/kernel_mdh_sparse > mdh_ncu.txt 2>&1
tail -20 mdh_ncu.txt

echo
echo "=== ppcg (matrix-free) ==="
$NCU --target-processes all --launch-skip 5 --launch-count 1 \
  --kernel-name kernel0 --metrics "$METRICS" \
  ../3_ppcg/kernel_ppcg_sparse > ppcg_ncu.txt 2>&1
tail -20 ppcg_ncu.txt

echo
echo "=== cusparse (true CSR, vendor library) ==="
$NCU --target-processes all --launch-skip 5 --launch-count 1 \
  --metrics "$METRICS" \
  ../4_cusparse/kernel_cusparse > cusparse_ncu.txt 2>&1
tail -25 cusparse_ncu.txt

echo
echo "All done. Full reports: sparse_ncu.txt, mdh_ncu.txt, ppcg_ncu.txt, cusparse_ncu.txt"
