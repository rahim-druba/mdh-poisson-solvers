#!/bin/bash
# RTX 5090: CSR-scalar + cuSPARSE vs hand-tuned vs MDH (untuned and tuned) at the large sizes, ONE session
# (reviewer issues #1/#4). PPCG is NOT included: it needs the ppcg binary to generate kernels for these sizes.
# Run on the RTX 5090 machine, after setting up the CUDA 12.9 environment
# (CUDA_HOME, CPATH, LIBRARY_PATH, CUDA_VISIBLE_DEVICES=1, ARCH=sm_120).
# Usage: nohup ./run_5090_formats.sh > raw/run_5090_formats.out 2>&1 < /dev/null &
# Output: raw/formats_5090/ ; status in raw/formats_5090/status.txt
set -e
cd "$(dirname "$0")"
CG="$(cd .. && pwd)"
ARCH=${ARCH:-sm_120}
OUT=${OUT:-raw/formats_5090}; mkdir -p "$OUT" bin
TUNE=${TUNE:-raw_5090}        # tuned configs from the earlier 5090 sweep (BEST_MDH_TUNED lines)
SIZES2D="512 2048 4096 8192 16384"
SIZES3D="64 128 256 512"
best_cfg() { grep BEST_MDH_TUNED "$TUNE/mdh_tune_$1_$2.txt" | sed 's/.*cfg=\[\(.*\)\]/\1/'; }
filt() { grep -viE "warning #68|warning #186|^generated|^$" || true; }

flags2d() { local S=$1 l1=$2 l2=$3 pc=$4 ca=$5
  echo "-DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=$ca -DCACHE_P_CB=$ca -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$S -DG_CB_SIZE_L_2=$S \
 -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=$pc -DP_CB_SIZE_L_2=1 \
 -DNUM_WG_L_1=$((S/l1)) -DNUM_WG_L_2=$((S/l2)) -DNUM_WI_L_1=$((l1/pc)) -DNUM_WI_L_2=$l2 -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0"; }
flags3d() { local M=$1 l1=$2 l2=$3 l3=$4 pc=$5 ca=$6
  echo "-DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=$ca -DCACHE_P_CB=$ca -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$M -DG_CB_SIZE_L_2=$M -DG_CB_SIZE_L_3=$M \
 -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 -DL_CB_SIZE_L_3=$l3 -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=$pc -DP_CB_SIZE_L_2=1 -DP_CB_SIZE_L_3=1 \
 -DNUM_WG_L_1=$((M/l1)) -DNUM_WG_L_2=$((M/l2)) -DNUM_WG_L_3=$((M/l3)) -DNUM_WI_L_1=$((l1/pc)) -DNUM_WI_L_2=$l2 -DNUM_WI_L_3=$l3 \
 -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_L_3=0"; }

for dim in 2d 3d; do
  if [ $dim = 2d ]; then SIZES=$SIZES2D; else SIZES=$SIZES3D; fi
  for S in $SIZES; do
    echo "$dim $S started $(date +%T)" >> "$OUT/status.txt"
    # GPU-state telemetry for this size
    nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader > "$OUT/gpu_before_${dim}_$S.txt"
    nvidia-smi --query-compute-apps=pid,name --format=csv,noheader > "$OUT/other_procs_${dim}_$S.txt" || true
    if [ $dim = 2d ]; then
      nvcc -O3 -std=c++14 -arch=$ARCH bench_csr.cu -o bin/csr_${dim}_$S -DDIM=2 -DROWS=$S -DCOLS=$S -lcusparse 2>&1 | filt
      read -r a b c d <<< "$(best_cfg 2d $S)"; FT=$(flags2d $S $a $b $c $d); FU=$(flags2d $S 16 16 1 0)
      MDHSRC="$CG/tables/matvec/cg_matvec_1.cu"; MK=cg_matvec_1; DEF="-DDIM=2 -DSIDE=$S"
    else
      nvcc -O3 -std=c++14 -arch=$ARCH bench_csr.cu -o bin/csr_${dim}_$S -DDIM=3 -DSIDE=$S -lcusparse 2>&1 | filt
      read -r a b c d e <<< "$(best_cfg 3d $S)"; FT=$(flags3d $S $a $b $c $d $e); FU=$(flags3d $S 8 8 8 1 0)
      MDHSRC="$CG/5_3d_extension/cg_matvec_3d_1.cu"; MK=cg_matvec_3d_1; DEF="-DDIM=3 -DSIDE=$S"
    fi
    nvcc -O3 -std=c++14 -arch=$ARCH bench_handtuned.cu "$MDHSRC" -o bin/htu_${dim}_$S $DEF -DWITH_MDH -DMDH_KERNEL=$MK $FU 2>&1 | filt
    nvcc -O3 -std=c++14 -arch=$ARCH bench_handtuned.cu "$MDHSRC" -o bin/htt_${dim}_$S $DEF -DWITH_MDH -DMDH_ONLY -DMDH_KERNEL=$MK $FT 2>&1 | filt
    # order inside one size: CSR/cuSPARSE, hand-tuned + MDH untuned, MDH tuned
    ./bin/csr_${dim}_$S  | tee "$OUT/csr_${dim}_$S.txt"
    ./bin/htu_${dim}_$S  | tee "$OUT/handtuned_untuned_${dim}_$S.txt"
    ./bin/htt_${dim}_$S  | tee "$OUT/mdh_tuned_${dim}_$S.txt"
    nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader > "$OUT/gpu_after_${dim}_$S.txt"
    echo "$dim $S done $(date +%T)" >> "$OUT/status.txt"
  done
done
echo ALL_DONE >> "$OUT/status.txt"
