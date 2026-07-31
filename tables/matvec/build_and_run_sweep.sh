#!/bin/bash
# Builds and runs the 4-way matvec benchmark at all 4 grid sizes.
set -e
cd "$(dirname "$0")"

build_and_run() {
  local size=$1 R=$2 C=$3 \
        lcb1=$4 lcb2=$5 nwg1=$6 nwg2=$7 nwi1=$8 nwi2=$9 \
        pbx=${10} pby=${11} pgx=${12} pgy=${13}

  echo "### N=$size (${R}x${C}) ###"
  nvcc -O3 bench_matvec.cu "$size/matvec_ppcg_src_kernel.cu" cg_matvec_1.cu \
    -o "bench_matvec_$size" \
    -DROWS=$R -DCOLS=$C \
    -DTYPE_T=float -DTYPE_TS=float \
    -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
    -DG_CB_RES_DEST_LEVEL=2 \
    -DG_CB_SIZE_L_1=$R -DG_CB_SIZE_L_2=$C \
    -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$lcb1 -DL_CB_SIZE_L_2=$lcb2 \
    -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1  -DP_CB_SIZE_L_2=1  \
    -DNUM_WG_L_1=$nwg1 -DNUM_WG_L_2=$nwg2 \
    -DNUM_WI_L_1=$nwi1 -DNUM_WI_L_2=$nwi2 \
    -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0 \
    -DPPCG_BLOCK_X=$pbx -DPPCG_BLOCK_Y=$pby -DPPCG_GRID_X=$pgx -DPPCG_GRID_Y=$pgy \
    -I"$size" \
    -lcusparse 2>&1 | grep -viE "warning #68|warning #186|^generated" || true

  "./bench_matvec_$size"
  echo
}

# size  R   C   L_CB1 L_CB2 NUM_WG1 NUM_WG2 NUM_WI1 NUM_WI2  PPCG_BLOCK_X PPCG_BLOCK_Y PPCG_GRID_X PPCG_GRID_Y
build_and_run 512  16 32  16 16  1 2  16 16   16 16  1 1
build_and_run 1024 32 32  16 16  2 2  16 16   16 32  1 1
build_and_run 2048 32 64  16 16  2 4  16 16   16 32  2 1
build_and_run 4096 64 64  16 16  4 4  16 16   16 32  2 2
