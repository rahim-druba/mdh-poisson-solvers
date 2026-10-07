#!/bin/bash
# Single-session harness at the PAPER's own sizes (reviewer issues #2/#3/#4).
# For every size it runs, back to back on the same GPU state:
#   1. CSR, cuSPARSE, PPCG, MDH with the paper's untouched config   (paper's bench_matvec*.cu)
#   2. the same four with MDH using its TUNED config               (tuned by mdh_tune_sweep.sh)
#   3. the hand-tuned matrix-free kernels (best of ~30 configs)     (bench_handtuned.cu)
# Sizes: 2D  512=16x32, 1024=32x32, 2048=32x64, 4096=64x64 ; 3D  8, 16, 24, 32.
# Usage: ./paper_sizes.sh [tune|run|all] [2d|3d|both]    (default: all both)
#   tune : only the MDH tuning sweeps (writes raw/paper_sizes/mdh_tune_*.txt)
#   run  : only the timed comparison (needs the tune files)
# Raw output: raw/paper_sizes/. Checks for GPU throttling before every size.
set -e
cd "$(dirname "$0")"
CG="$(cd .. && pwd)"
ARCH=${ARCH:-sm_86}
STAGE=${1:-all}; WHICH=${2:-both}
OUT=raw/paper_sizes; mkdir -p "$OUT" bin
export OUTDIR=$OUT

SIZES2D="16x32 32x32 32x64 64x64"
SIZES3D="8 16 24 32"
# paper's untuned PPCG launch params per 2D size (tables/matvec/build_and_run_sweep.sh)
declare -A PPCG2D=( [16x32]="16 16 1 1" [32x32]="16 32 1 1" [32x64]="16 32 2 1" [64x64]="16 32 2 2" )

check_gpu() {
  local bad; bad=$(nvidia-smi -q -d PERFORMANCE | grep -E "SW Power Cap|SW Thermal|HW Slowdown|HW Thermal|HW Power" | grep -v "Not Active" || true)
  if [ -n "$bad" ]; then echo "GPU throttling active, aborting timings:"; echo "$bad"; exit 1; fi
  [ "$(cat /sys/class/power_supply/ACAD/online 2>/dev/null || echo 1)" = 1 ] || { echo "charger not plugged in"; exit 1; }
}

best_cfg() { grep BEST_MDH_TUNED "$OUT/mdh_tune_$1_$2.txt" | sed 's/.*cfg=\[\(.*\)\]/\1/'; }

# -D flags for a tuned 2D config: R C l1 l2 pc ca
flags2d() { local R=$1 C=$2 l1=$3 l2=$4 pc=$5 ca=$6
  echo "-DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=$ca -DCACHE_P_CB=$ca \
 -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$R -DG_CB_SIZE_L_2=$C \
 -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 \
 -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=$pc -DP_CB_SIZE_L_2=1 \
 -DNUM_WG_L_1=$((R/l1)) -DNUM_WG_L_2=$((C/l2)) -DNUM_WI_L_1=$((l1/pc)) -DNUM_WI_L_2=$l2 \
 -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0"; }
# 3D: M l1 l2 l3 pc ca
flags3d() { local M=$1 l1=$2 l2=$3 l3=$4 pc=$5 ca=$6
  echo "-DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=$ca -DCACHE_P_CB=$ca \
 -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$M -DG_CB_SIZE_L_2=$M -DG_CB_SIZE_L_3=$M \
 -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 -DL_CB_SIZE_L_3=$l3 \
 -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=$pc -DP_CB_SIZE_L_2=1 -DP_CB_SIZE_L_3=1 \
 -DNUM_WG_L_1=$((M/l1)) -DNUM_WG_L_2=$((M/l2)) -DNUM_WG_L_3=$((M/l3)) \
 -DNUM_WI_L_1=$((l1/pc)) -DNUM_WI_L_2=$l2 -DNUM_WI_L_3=$l3 \
 -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_L_3=0"; }

filt() { grep -viE "warning #68|warning #186|warning #177|^generated|^$|Remark" || true; }


PT=$CG/ppcg_tuning
ppcg_cfg() { grep BEST_PPCG_TUNED "$PT/raw/paper/ppcg_tune_$1_$2.txt" | sed 's/.*cfg=\[\(.*\)\]/\1/'; }   # "tile | block"
ppcg_tuned_run() {   # dim size(R C or M) tag
  local dim=$1 cfg tile block d
  cfg=$(ppcg_cfg $dim $2); tile=${cfg%% |*}; block=${cfg##*| }
  d=$PWD/bin/ppcg_gen_$3; rm -rf $d
  if [ $dim = 2d ]; then "$PT/ppcg_gen.sh" 2d ${2%x*} ${2#*x} $d "$tile" "$block"; DEF="-DDIM=2 -DROWS=${2%x*} -DCOLS=${2#*x}"
  else "$PT/ppcg_gen.sh" 3d $2 - $d "$tile" "$block"; DEF="-DDIM=3 -DSIDE=$2"; fi
  read bx by bz gx gy < $d/dims.txt
  nvcc -O3 -std=c++14 -arch=$ARCH "$PT/bench_ppcg.cu" $d/kernel.cu -I $d -o $d/b $DEF -DBX=$bx -DBY=$by -DBZ=$bz -DGX=$gx -DGY=$gy 2>&1 | filt
  { echo "### PPCG tuned cfg=[$cfg]"; $d/b; } | tee $OUT/ppcg_tuned_$3.txt
}

# ---------------------------------------------------------------- tuning
if [ "$STAGE" = tune ] || [ "$STAGE" = all ]; then
  [ "$WHICH" = 3d ] || for s in $SIZES2D; do check_gpu; ./mdh_tune_sweep.sh 2d $s; done
  [ "$WHICH" = 2d ] || for m in $SIZES3D; do check_gpu; ./mdh_tune_sweep.sh 3d $m; done
fi

# ---------------------------------------------------------------- timed runs
if [ "$STAGE" = run ] || [ "$STAGE" = all ]; then
  if [ "$WHICH" != 3d ]; then
    for s in $SIZES2D; do
      R=${s%x*}; C=${s#*x}; N=$((R*C)); D="$CG/tables/matvec/$N"
      check_gpu
      read -r l1 l2 pc ca <<< "$(best_cfg 2d $s)"
      read -r PBX PBY PGX PGY <<< "${PPCG2D[$s]}"
      PP="-DPPCG_BLOCK_X=$PBX -DPPCG_BLOCK_Y=$PBY -DPPCG_GRID_X=$PGX -DPPCG_GRID_Y=$PGY"
      for variant in untuned tuned; do
        if [ $variant = untuned ]; then F=$(flags2d $R $C 16 16 1 0); else F=$(flags2d $R $C $l1 $l2 $pc $ca); fi
        nvcc -O3 -std=c++14 -arch=$ARCH -DROWS=$R -DCOLS=$C $F $PP -I"$D" \
          "$CG/tables/matvec/bench_matvec.cu" "$D/matvec_ppcg_src_kernel.cu" "$CG/tables/matvec/cg_matvec_1.cu" \
          -o bin/ps2d_${variant}_$N -lcusparse 2>&1 | filt
        { echo "### 2D N=$N (${R}x${C}) MDH $variant $( [ $variant = tuned ] && echo "cfg=[$l1 $l2 $pc $ca]")"; ./bin/ps2d_${variant}_$N; } | tee $OUT/matvec_2d_${N}_$variant.txt
      done
      nvcc -O3 -std=c++14 -arch=$ARCH bench_handtuned.cu -o bin/ps2d_ht_$N -DDIM=2 -DROWS=$R -DCOLS=$C 2>&1 | filt
      ./bin/ps2d_ht_$N | tee $OUT/handtuned_2d_${N}.txt
      ppcg_tuned_run 2d $s 2d_$N
    done
  fi
  if [ "$WHICH" != 2d ]; then
    for M in $SIZES3D; do
      N=$((M*M*M)); D="$CG/5_3d_extension/sizes/$M"
      check_gpu
      read -r l1 l2 l3 pc ca <<< "$(best_cfg 3d $M)"
      for variant in untuned tuned; do
        if [ $variant = untuned ]; then F=$(flags3d $M 8 8 8 1 0); else F=$(flags3d $M $l1 $l2 $l3 $pc $ca); fi
        nvcc -O3 -std=c++14 -arch=$ARCH -DM=$M -I"$D" "$CG/5_3d_extension/bench_matvec_3d.cu" \
          "$CG/5_3d_extension/cg_matvec_3d_1.cu" "$D/cg_matvec_3d_ppcg_src_kernel.cu" \
          -o bin/ps3d_${variant}_$M $F -lcusparse 2>&1 | filt
        { echo "### 3D M=$M N=$N MDH $variant $( [ $variant = tuned ] && echo "cfg=[$l1 $l2 $l3 $pc $ca]")"; ./bin/ps3d_${variant}_$M; } | tee $OUT/matvec_3d_${M}_$variant.txt
      done
      nvcc -O3 -std=c++14 -arch=$ARCH bench_handtuned.cu -o bin/ps3d_ht_$M -DDIM=3 -DSIDE=$M 2>&1 | filt
      ./bin/ps3d_ht_$M | tee $OUT/handtuned_3d_${M}.txt
      ppcg_tuned_run 3d $M 3d_$M
    done
  fi
fi
