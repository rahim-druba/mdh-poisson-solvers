#!/bin/bash
# Builds and runs the hand-tuned matrix-free baseline vs the existing MDH kernel.
# Usage: ./build_and_run.sh            (all sizes)
#        ./build_and_run.sh 2d 1024    (one size)
# Raw output goes to raw/<dim>_<side>.txt (kept for the reproducibility
# requirement, reviewer issue #6). Reuses the MDH kernels already in the repo
# (same -D macros as tables/matvec and 5_3d_extension, tile 16x16 / 8x8x8).
set -e
cd "$(dirname "$0")"
mkdir -p raw bin
CG="$(cd .. && pwd)"
ARCH=${ARCH:-sm_86}      # RTX 3050 Laptop = sm_86; use sm_120 for RTX 5080/5090 (CUDA >= 12.8)

build_run_2d() {
  local S=$1
  nvcc -O3 -std=c++14 -arch=$ARCH bench_handtuned.cu "$CG/tables/matvec/cg_matvec_1.cu" -o bin/ht_2d_$S \
    -DDIM=2 -DSIDE=$S -DWITH_MDH -DMDH_KERNEL=cg_matvec_1 \
    -DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
    -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$S -DG_CB_SIZE_L_2=$S \
    -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=16 -DL_CB_SIZE_L_2=16 \
    -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1 \
    -DNUM_WG_L_1=$((S/16)) -DNUM_WG_L_2=$((S/16)) -DNUM_WI_L_1=16 -DNUM_WI_L_2=16 \
    -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0 2>&1 | grep -viE "warning #68|warning #186|^generated|^$" || true
  ./bin/ht_2d_$S | tee raw/2d_$S.txt
}

build_run_3d() {
  local M=$1
  nvcc -O3 -std=c++14 -arch=$ARCH bench_handtuned.cu "$CG/5_3d_extension/cg_matvec_3d_1.cu" -o bin/ht_3d_$M \
    -DDIM=3 -DSIDE=$M -DWITH_MDH -DMDH_KERNEL=cg_matvec_3d_1 \
    -DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
    -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$M -DG_CB_SIZE_L_2=$M -DG_CB_SIZE_L_3=$M \
    -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=8 -DL_CB_SIZE_L_2=8 -DL_CB_SIZE_L_3=8 \
    -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1 -DP_CB_SIZE_L_3=1 \
    -DNUM_WG_L_1=$((M/8)) -DNUM_WG_L_2=$((M/8)) -DNUM_WG_L_3=$((M/8)) \
    -DNUM_WI_L_1=8 -DNUM_WI_L_2=8 -DNUM_WI_L_3=8 \
    -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_L_3=0 2>&1 | grep -viE "warning #68|warning #186|^generated|^$" || true
  ./bin/ht_3d_$M | tee raw/3d_$M.txt
}

if [ $# -eq 2 ]; then
  [ "$1" = 2d ] && build_run_2d $2 || build_run_3d $2
else
  for S in 64 512 2048 4096; do build_run_2d $S; echo; done
  for M in 16 32 64 128 256; do build_run_3d $M; echo; done
fi
