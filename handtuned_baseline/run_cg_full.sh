#!/bin/bash
# Fully MDH-generated GPU-resident CG vs hand-written variants, large sizes, fixed iterations (cg_resident.cu).
# Per size, five variants in one session (ms per iteration):
#   D  hand-tuned stencil, FUSED hand-written vector kernels         (MV=0, VEC=-1)   [fusion reference]
#   E  MDH stencil,        FUSED hand-written vector kernels         (MV=3, VEC=-1)
#   A  hand-tuned stencil, UNFUSED hand-written dot/axpy             (MV=0, VEC=0)
#   B  MDH stencil,        UNFUSED hand-written dot/axpy             (MV=3, VEC=0)
#   C  MDH stencil + MDH-generated dot product + MDH-generated axpy  (MV=3, VEC=1)   [fully generated]
#   F  hand-written CSR kernel, fused hand-written vector kernels      (MV=1, VEC=-1)
#   G  cuSPARSE SpMV (CSR),     fused hand-written vector kernels      (MV=2, VEC=-1)
# env: ARCH, CFG (table "dim size BX BY CY"), OUT, MDHTUNE (dir with mdh_tune_<dim>_<s>.txt), DOTMAX (max partial sums)
set -e
cd "$(dirname "$0")"; CG="$(cd .. && pwd)"
ARCH=${ARCH:-sm_86}; CFG=${CFG:?}; OUT=${OUT:?}; MDHTUNE=${MDHTUNE:?}; DOTMAX=${DOTMAX:-4096}
mkdir -p "$OUT" bin/full; : > "$OUT/status.txt"
best_mdh() { grep BEST_MDH_TUNED "$MDHTUNE/mdh_tune_$1_$2.txt" | sed 's/.*cfg=\[\(.*\)\]/\1/'; }
COM="-DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=0 -DCACHE_P_CB=0 -DG_CB_RES_DEST_LEVEL=2 -DL_CB_RES_DEST_LEVEL=1 -DP_CB_RES_DEST_LEVEL=0"
st2d() { local S=$1 l1=$2 l2=$3 pc=$4 ca=$5
  echo "-DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=$ca -DCACHE_P_CB=$ca -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$S -DG_CB_SIZE_L_2=$S -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=$pc -DP_CB_SIZE_L_2=1 -DNUM_WG_L_1=$((S/l1)) -DNUM_WG_L_2=$((S/l2)) -DNUM_WI_L_1=$((l1/pc)) -DNUM_WI_L_2=$l2 -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0"; }
st3d() { local M=$1 l1=$2 l2=$3 l3=$4 pc=$5 ca=$6
  echo "-DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=$ca -DCACHE_P_CB=$ca -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$M -DG_CB_SIZE_L_2=$M -DG_CB_SIZE_L_3=$M -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 -DL_CB_SIZE_L_3=$l3 -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=$pc -DP_CB_SIZE_L_2=1 -DP_CB_SIZE_L_3=1 -DNUM_WG_L_1=$((M/l1)) -DNUM_WG_L_2=$((M/l2)) -DNUM_WG_L_3=$((M/l3)) -DNUM_WI_L_1=$((l1/pc)) -DNUM_WI_L_2=$l2 -DNUM_WI_L_3=$l3 -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_L_3=0"; }
while read dim s bx by cy; do
  echo "$dim $s started $(date +%T)" >> "$OUT/status.txt"
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader > "$OUT/gpu_before_${dim}_$s.txt"
  if [ $dim = 2d ]; then N=$((s*s)); DEF="-DDIM=2 -DROWS=$s -DCOLS=$s"; read -r a b c d <<< "$(best_mdh 2d $s)"; FST=$(st2d $s $a $b $c $d); MSRC="$CG/tables/matvec/cg_matvec_1.cu"; MK=cg_matvec_1
    STH="-DST_NUM_WG_L_1=$((s/a)) -DST_NUM_WG_L_2=$((s/b)) -DST_NUM_WI_L_1=$((a/c)) -DST_NUM_WI_L_2=$b"
  else N=$((s*s*s)); DEF="-DDIM=3 -DSIDE=$s"; read -r a b c d e <<< "$(best_mdh 3d $s)"; FST=$(st3d $s $a $b $c $d $e); MSRC="$CG/5_3d_extension/cg_matvec_3d_1.cu"; MK=cg_matvec_3d_1
    STH="-DST_NUM_WG_L_1=$((s/a)) -DST_NUM_WG_L_2=$((s/b)) -DST_NUM_WG_L_3=$((s/c)) -DST_NUM_WI_L_1=$((a/d)) -DST_NUM_WI_L_2=$b -DST_NUM_WI_L_3=$c"; fi
  DI=256; DW=$((N/DI)); [ $DW -gt $DOTMAX ] && DW=$DOTMAX
  AI=256; AW=$((N/AI))
  FDOT="$COM -DG_CB_SIZE_L_1=1 -DL_CB_SIZE_L_1=1 -DP_CB_SIZE_L_1=1 -DNUM_WG_L_1=1 -DNUM_WI_L_1=1 -DOCL_DIM_L_1=1 -DOCL_DIM_R_1=0 -DG_CB_SIZE_R_1=$N -DL_CB_SIZE_R_1=$((N/DW)) -DP_CB_SIZE_R_1=$((N/DW/DI)) -DNUM_WG_R_1=$DW -DNUM_WI_R_1=$DI"
  FAX="$COM -DG_CB_SIZE_L_1=$N -DL_CB_SIZE_L_1=$AI -DP_CB_SIZE_L_1=1 -DNUM_WG_L_1=$AW -DNUM_WI_L_1=$AI -DOCL_DIM_L_1=0"
  T=bin/full/${dim}_$s; mkdir -p $T
  nvcc -O3 -std=c++14 -arch=$ARCH -c "$MSRC" $FST -o $T/st.o 2>&1 | grep -i " error" || true
  nvcc -O3 -std=c++14 -arch=$ARCH -c mdh_vec/dot_1.cu $FDOT -o $T/dot.o 2>&1 | grep -i " error" || true
  nvcc -O3 -std=c++14 -arch=$ARCH -c mdh_vec/axpy_1.cu $FAX -o $T/axpy.o 2>&1 | grep -i " error" || true
  H="$DEF -DBX=$bx -DBY=$by -DCY=$cy -DMDH_KERNEL=$MK $STH -DDOT_WG=$DW -DDOT_WI=$DI -DAX_WG=$AW -DAX_WI=$AI"
  build() { nvcc -O3 -std=c++14 -arch=$ARCH cg_resident.cu "${@:2}" $H -lcusparse -o $T/$1 2>&1 | grep -i " error" || true; }
  build D -DMV=0 -DVEC=-1;   build E $T/st.o -DMV=3 -DVEC=-1
  build A -DMV=0 -DVEC=0;    build B $T/st.o -DMV=3 -DVEC=0
  build C $T/st.o $T/dot.o $T/axpy.o -DMV=3 -DVEC=1
  build F -DMV=1 -DVEC=-1;   build G -DMV=2 -DVEC=-1
  echo "# hand-tuned reg ${bx}x${by} c${cy}; MDH stencil cfg [$(best_mdh $dim $s)]; dot WG=$DW WI=$DI; axpy WI=$AI; variants D fused-hw, E fused-hw+MDH-stencil, A unfused-hw, B unfused-hw+MDH-stencil, C fully-MDH" > "$OUT/cgfull_${dim}_$s.txt"
  for v in D E A B C F G; do echo -n "$v " | tee -a "$OUT/cgfull_${dim}_$s.txt"; ./$T/$v | grep "^cg-resident" | tee -a "$OUT/cgfull_${dim}_$s.txt"; done
  echo "$dim $s done $(date +%T)" >> "$OUT/status.txt"
done < "$CFG"
echo ALL_DONE >> "$OUT/status.txt"
