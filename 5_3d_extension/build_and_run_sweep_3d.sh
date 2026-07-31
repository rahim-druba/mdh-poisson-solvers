#!/bin/bash
# Builds and runs all 4 solvers (sparse/mdh/ppcg/cusparse) at 4 cube sizes
# (8/16/24/32 -> N=512/4096/13824/32768) -- the 3D counterpart of
# ../tables/full_cg/build_and_run_sweep.sh, same protocol (10 runs/9
# averaged). PPCG kernels are pre-generated per size under sizes/<M>/
# (regeneration needs the built ppcg binary, not scripted here -- see
# results.md). MDH's kernel (cg_matvec_3d_1.cu) is size-agnostic, just
# recompiled with different -D tile macros per size.
set -e
cd "$(dirname "$0")"

build_size() {
  local M=$1
  local L_CB=8
  local NUM_WG=$((M / L_CB))
  local NUM_WI=$L_CB

  nvcc -O3 -DM=$M kernel_sparse_3d.cu -o "sparse_$M" 2>&1 | grep -v "^$" || true

  nvcc -O3 kernel_mdh_3d.cu cg_matvec_3d_1.cu -o "mdh_$M" \
    -DTYPE_T=float -DTYPE_TS=float \
    -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
    -DG_CB_RES_DEST_LEVEL=2 \
    -DG_CB_SIZE_L_1=$M -DG_CB_SIZE_L_2=$M -DG_CB_SIZE_L_3=$M \
    -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$L_CB -DL_CB_SIZE_L_2=$L_CB -DL_CB_SIZE_L_3=$L_CB \
    -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1 -DP_CB_SIZE_L_3=1 \
    -DNUM_WG_L_1=$NUM_WG -DNUM_WG_L_2=$NUM_WG -DNUM_WG_L_3=$NUM_WG \
    -DNUM_WI_L_1=$NUM_WI -DNUM_WI_L_2=$NUM_WI -DNUM_WI_L_3=$NUM_WI \
    -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_L_3=0 2>&1 | grep -viE "warning #68|warning #186|^generated" || true

  nvcc -O3 -DM=$M -I "sizes/$M" kernel_ppcg_3d.cu "sizes/$M/cg_matvec_3d_ppcg_src_kernel.cu" \
    -o "ppcg_$M" 2>&1 | grep -v "warning #177" || true

  nvcc -O3 -DM=$M kernel_cusparse_3d.cu -o "cusparse_$M" -lcusparse 2>&1 | grep -v "^$" || true
}

for M in 8 16 24 32; do
  build_size $M
done

echo "All 16 binaries built."
echo

printf "%-8s %-10s %8s %10s %14s %14s\n" "size" "method" "runs" "avg ms" "iterations" "max err"
printf -- "----------------------------------------------------------------------\n"

for M in 8 16 24 32; do
  N=$((M * M * M))
  for method in sparse mdh ppcg cusparse; do
    bin="./${method}_${M}"
    times=()
    iters=""; err=""
    for run in $(seq 1 10); do
      out=$("$bin")
      t=$(echo "$out" | grep -oP 'Time taken for main loop: \K[0-9.]+')
      it=$(echo "$out" | grep -oP 'Converged in \K[0-9]+')
      e=$(echo "$out" | grep -oP 'Max abs error vs analytical solution: \K[0-9.eE+-]+')
      if [ "$run" -gt 1 ]; then times+=("$t"); fi
      iters="$it"; err="$e"
    done
    avg=$(printf '%s\n' "${times[@]}" | LC_NUMERIC=C awk '{s+=$1} END {printf "%.4f", s/NR}')
    printf "%-8s %-10s %8d %10s %14s %14s\n" "N=$N (${M}^3)" "$method" "${#times[@]}" "$avg" "$iters" "$err"
  done
done
