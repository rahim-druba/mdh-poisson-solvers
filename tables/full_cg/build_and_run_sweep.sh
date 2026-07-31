#!/bin/bash
# Builds and runs all 4 CG solvers (sparse/mdh/ppcg/cusparse) at all 4 sizes.
# Reuses the per-size MDH/PPCG kernels already generated for the matvec table.
set -e
cd "$(dirname "$0")"
MV=../matvec

build_size() {
  local size=$1 R=$2 C=$3 \
        lcb1=$4 lcb2=$5 nwg1=$6 nwg2=$7 nwi1=$8 nwi2=$9 \
        pbx=${10} pby=${11} pgx=${12} pgy=${13}

  nvcc -O3 kernel_sparse_cg.cu -o "sparse_$size" -DROWS=$R -DCOLS=$C 2>&1 | grep -v "^$" || true

  nvcc -O3 kernel_mdh_cg.cu "$MV/cg_matvec_1.cu" -o "mdh_$size" \
    -DROWS=$R -DCOLS=$C \
    -DTYPE_T=float -DTYPE_TS=float \
    -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
    -DG_CB_RES_DEST_LEVEL=2 \
    -DG_CB_SIZE_L_1=$R -DG_CB_SIZE_L_2=$C \
    -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$lcb1 -DL_CB_SIZE_L_2=$lcb2 \
    -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1  -DP_CB_SIZE_L_2=1  \
    -DNUM_WG_L_1=$nwg1 -DNUM_WG_L_2=$nwg2 \
    -DNUM_WI_L_1=$nwi1 -DNUM_WI_L_2=$nwi2 \
    -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0 2>&1 | grep -viE "warning #68|warning #186|^generated" || true

  nvcc -O3 kernel_ppcg_cg.cu "$MV/$size/matvec_ppcg_src_kernel.cu" -o "ppcg_$size" \
    -DROWS=$R -DCOLS=$C \
    -DPPCG_BLOCK_X=$pbx -DPPCG_BLOCK_Y=$pby -DPPCG_GRID_X=$pgx -DPPCG_GRID_Y=$pgy \
    -I"$MV/$size" 2>&1 | grep -v "^$" || true

  nvcc -O3 kernel_cusparse_cg.cu -o "cusparse_$size" -DROWS=$R -DCOLS=$C -lcusparse 2>&1 | grep -v "^$" || true
}

# size  R   C   L_CB1 L_CB2 NUM_WG1 NUM_WG2 NUM_WI1 NUM_WI2  PPCG_BLOCK_X PPCG_BLOCK_Y PPCG_GRID_X PPCG_GRID_Y
build_size 512  16 32  16 16  1 2  16 16   16 16  1 1
build_size 1024 32 32  16 16  2 2  16 16   16 32  1 1
build_size 2048 32 64  16 16  2 4  16 16   16 32  2 1
build_size 4096 64 64  16 16  4 4  16 16   16 32  2 2

echo "All 16 binaries built."
echo
printf "%-8s %-10s %8s %10s %14s %14s\n" "size" "method" "runs" "avg ms" "iterations" "max err"
printf -- "----------------------------------------------------------------------\n"

for size in 512 1024 2048 4096; do
  for method in sparse mdh ppcg cusparse; do
    bin="./${method}_${size}"
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
    printf "%-8s %-10s %8d %10s %14s %14s\n" "$size" "$method" "${#times[@]}" "$avg" "$iters" "$err"
  done
done
