#!/bin/bash
# GPU-resident CG at large sizes, five matvec variants in the SAME loop (see cg_resident.cu): hand-tuned matrix-free (0),
# MDH tuned (3), PPCG tuned (4), hand-written CSR (1), cuSPARSE (2). Fixed ITERS iterations, ms per iteration.
# env: ARCH, CFG (hand-tuned reg config table: "dim size BX BY CY"), OUT, MDHTUNE (dir with mdh_tune_<dim>_<s>.txt),
#      PPCGRES (dir with ppcg_<dim>_<s>.txt holding BEST_PPCG_TUNED), CANDS (dir with <dim>_<s>/cfgs.txt and c<i>/ kernels)
set -e
cd "$(dirname "$0")"; CG="$(cd .. && pwd)"
ARCH=${ARCH:-sm_86}; CFG=${CFG:?}; OUT=${OUT:?}; MDHTUNE=${MDHTUNE:?}; PPCGRES=${PPCGRES:?}; CANDS=${CANDS:?}
mkdir -p "$OUT" bin; : > "$OUT/status.txt"
best_mdh() { grep BEST_MDH_TUNED "$MDHTUNE/mdh_tune_$1_$2.txt" | sed 's/.*cfg=\[\(.*\)\]/\1/'; }
best_ppcg() { grep BEST_PPCG_TUNED "$PPCGRES/ppcg_$1_$2.txt" | sed 's/.*cfg=\[\(.*\)\]/\1/'; }
flags2d() { local S=$1 l1=$2 l2=$3 pc=$4 ca=$5
  echo "-DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=$ca -DCACHE_P_CB=$ca -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$S -DG_CB_SIZE_L_2=$S \
 -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=$pc -DP_CB_SIZE_L_2=1 \
 -DNUM_WG_L_1=$((S/l1)) -DNUM_WG_L_2=$((S/l2)) -DNUM_WI_L_1=$((l1/pc)) -DNUM_WI_L_2=$l2 -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0"; }
flags3d() { local M=$1 l1=$2 l2=$3 l3=$4 pc=$5 ca=$6
  echo "-DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=$ca -DCACHE_P_CB=$ca -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$M -DG_CB_SIZE_L_2=$M -DG_CB_SIZE_L_3=$M \
 -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 -DL_CB_SIZE_L_3=$l3 -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=$pc -DP_CB_SIZE_L_2=1 -DP_CB_SIZE_L_3=1 \
 -DNUM_WG_L_1=$((M/l1)) -DNUM_WG_L_2=$((M/l2)) -DNUM_WG_L_3=$((M/l3)) -DNUM_WI_L_1=$((l1/pc)) -DNUM_WI_L_2=$l2 -DNUM_WI_L_3=$l3 \
 -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_L_3=0"; }
while read dim s bx by cy; do
  echo "$dim $s started $(date +%T)" >> "$OUT/status.txt"
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader > "$OUT/gpu_before_${dim}_$s.txt"
  if [ $dim = 2d ]; then DEF="-DDIM=2 -DROWS=$s -DCOLS=$s"; read -r a b c d <<< "$(best_mdh 2d $s)"; MF=$(flags2d $s $a $b $c $d); MSRC="$CG/tables/matvec/cg_matvec_1.cu"; MK=cg_matvec_1
  else DEF="-DDIM=3 -DSIDE=$s"; read -r a b c d e <<< "$(best_mdh 3d $s)"; MF=$(flags3d $s $a $b $c $d $e); MSRC="$CG/5_3d_extension/cg_matvec_3d_1.cu"; MK=cg_matvec_3d_1; fi
  cfgp=$(best_ppcg $dim $s); tile=${cfgp%% |*}; block=${cfgp##*| }
  idx=$(awk -v t="$tile" -v b="$block" '$2==t && $3==b {print $1}' "$CANDS/${dim}_$s/cfgs.txt" | head -1)
  PD="$CANDS/${dim}_$s/c$idx"; read pbx pby pbz pgx pgy < "$PD/dims.txt"
  echo "# hand-tuned reg ${bx}x${by} c${cy}; MDH tuned cfg [$(best_mdh $dim $s)]; PPCG tuned cfg [$cfgp] (candidate $idx)" > "$OUT/cgres_${dim}_$s.txt"
  B="-DBX=$bx -DBY=$by -DCY=$cy"
  nvcc -O3 -std=c++14 -arch=$ARCH cg_resident.cu -o bin/cgr_${dim}_${s}_0 $DEF $B -DMV=0 -lcusparse 2>&1 | grep -iE "error" || true
  nvcc -O3 -std=c++14 -arch=$ARCH cg_resident.cu "$MSRC" -o bin/cgr_${dim}_${s}_3 $DEF $B -DMV=3 -DMDH_KERNEL=$MK $MF -lcusparse 2>&1 | grep -iE "error" || true
  nvcc -O3 -std=c++14 -arch=$ARCH cg_resident.cu "$PD/kernel.cu" -I "$PD" -o bin/cgr_${dim}_${s}_4 $DEF $B -DMV=4 -DPBX=$pbx -DPBY=$pby -DPBZ=$pbz -DPGX=$pgx -DPGY=$pgy -lcusparse 2>&1 | grep -iE "error" || true
  nvcc -O3 -std=c++14 -arch=$ARCH cg_resident.cu -o bin/cgr_${dim}_${s}_1 $DEF $B -DMV=1 -lcusparse 2>&1 | grep -iE "error" || true
  nvcc -O3 -std=c++14 -arch=$ARCH cg_resident.cu -o bin/cgr_${dim}_${s}_2 $DEF $B -DMV=2 -lcusparse 2>&1 | grep -iE "error" || true
  for mv in 0 3 4 1 2; do ./bin/cgr_${dim}_${s}_$mv | tee -a "$OUT/cgres_${dim}_$s.txt"; done
  echo "$dim $s done $(date +%T)" >> "$OUT/status.txt"
done < "$CFG"
echo ALL_DONE >> "$OUT/status.txt"
