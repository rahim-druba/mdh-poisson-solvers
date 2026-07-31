#!/bin/bash
# Builds and runs the 4-way dense matmul benchmark at all 4 sizes.
set -e
cd "$(dirname "$0")"

build_and_run() {
  local size=$1 pbx=$2 pby=$3 pgx=$4 pgy=$5

  echo "### N=$size ###"
  nvcc -O3 bench_matmul.cu "$size/matmul_ppcg_src_kernel.cu" gemm_1.cu -o "bench_matmul_$size" \
    -DGEMM_N=$size \
    -DTYPE_T=float -DTYPE_TS=float \
    -DCACHE_L_CB=1 -DCACHE_P_CB=1 \
    -DG_CB_RES_DEST_LEVEL=2 -DL_CB_RES_DEST_LEVEL=1 -DP_CB_RES_DEST_LEVEL=0 \
    -DG_CB_SIZE_L_1=$size -DG_CB_SIZE_L_2=$size -DG_CB_SIZE_R_1=$size \
    -DL_CB_SIZE_L_1=64 -DL_CB_SIZE_L_2=64 -DL_CB_SIZE_R_1=16 \
    -DP_CB_SIZE_L_1=4 -DP_CB_SIZE_L_2=4 -DP_CB_SIZE_R_1=1 \
    -DNUM_WG_L_1=$((size/64)) -DNUM_WG_L_2=$((size/64)) -DNUM_WG_R_1=1 \
    -DNUM_WI_L_1=16 -DNUM_WI_L_2=16 -DNUM_WI_R_1=1 \
    -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_R_1=0 \
    -DPPCG_BLOCK_X=$pbx -DPPCG_BLOCK_Y=$pby -DPPCG_GRID_X=$pgx -DPPCG_GRID_Y=$pgy \
    -I"$size" -lcublas 2>&1 | grep -viE "warning #68|warning #186|^generated" || true

  "./bench_matmul_$size"
  echo
}

# size  PPCG_BLOCK_X PPCG_BLOCK_Y PPCG_GRID_X PPCG_GRID_Y
build_and_run 512  16 32 256 16
build_and_run 1024 16 32 256 32
build_and_run 2048 16 32 256 64
build_and_run 4096 16 32 256 128
