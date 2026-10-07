#!/bin/bash
# Full CG solves at the PAPER's sizes, one session (reviewer issues #3/#4): the paper's four solvers
# (CSR, MDH untuned, PPCG default, cuSPARSE) + MDH with the TUNED config + hand-tuned matrix-free CG
# (host-loop version = same structure as the paper's; device-resident version = best-case solver).
# Protocol as the paper's tables: 10 runs per solver, first dropped, 9 averaged ("Time taken for main loop", clock()).
# Needs raw/paper_sizes/mdh_tune_*.txt (tuned configs) and raw/paper_sizes/handtuned_*.txt (best reg config).
# Usage: ./paper_cg.sh [2d|3d|both]       Output: raw/paper_cg/results.txt + per-run logs
set -e
cd "$(dirname "$0")"
CG="$(cd .. && pwd)"; ARCH=${ARCH:-sm_86}; WHICH=${1:-both}
PS=raw/paper_sizes; OUT=raw/paper_cg; mkdir -p "$OUT" bin
SIZES2D="16x32 32x32 32x64 64x64"; SIZES3D="8 16 24 32"
declare -A PPCG2D=( [16x32]="16 16 1 1" [32x32]="16 32 1 1" [32x64]="16 32 2 1" [64x64]="16 32 2 2" )
filt() { grep -viE "warning #68|warning #186|warning #177|^generated|^$|Remark" || true; }
check_gpu() {
  local bad; bad=$(nvidia-smi -q -d PERFORMANCE | grep -E "SW Power Cap|SW Thermal|HW Slowdown|HW Thermal|HW Power" | grep -v "Not Active" || true)
  [ -z "$bad" ] || { echo "GPU throttling active, aborting:"; echo "$bad"; exit 1; }
  [ "$(cat /sys/class/power_supply/ACAD/online 2>/dev/null || echo 1)" = 1 ] || { echo "charger not plugged in"; exit 1; }
}
best_cfg() { grep BEST_MDH_TUNED "$PS/mdh_tune_$1_$2.txt" | sed 's/.*cfg=\[\(.*\)\]/\1/'; }
best_reg() { grep -E "^  best reg" "$PS/handtuned_$1_$2.txt" | sed -E 's/.*reg +([0-9]+)x([0-9]+) +c[yi]([0-9]+).*/\1 \2 \3/'; }

flags2d() { local R=$1 C=$2 l1=$3 l2=$4 pc=$5 ca=$6
  echo "-DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=$ca -DCACHE_P_CB=$ca -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$R -DG_CB_SIZE_L_2=$C \
 -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=$pc -DP_CB_SIZE_L_2=1 \
 -DNUM_WG_L_1=$((R/l1)) -DNUM_WG_L_2=$((C/l2)) -DNUM_WI_L_1=$((l1/pc)) -DNUM_WI_L_2=$l2 -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0"; }
flags3d() { local M=$1 l1=$2 l2=$3 l3=$4 pc=$5 ca=$6
  echo "-DTYPE_T=float -DTYPE_TS=float -DCACHE_L_CB=$ca -DCACHE_P_CB=$ca -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$M -DG_CB_SIZE_L_2=$M -DG_CB_SIZE_L_3=$M \
 -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 -DL_CB_SIZE_L_3=$l3 -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=$pc -DP_CB_SIZE_L_2=1 -DP_CB_SIZE_L_3=1 \
 -DNUM_WG_L_1=$((M/l1)) -DNUM_WG_L_2=$((M/l2)) -DNUM_WG_L_3=$((M/l3)) -DNUM_WI_L_1=$((l1/pc)) -DNUM_WI_L_2=$l2 -DNUM_WI_L_3=$l3 \
 -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_L_3=0"; }

PT=$CG/ppcg_tuning
ppcgt_gen() {   # dim sizekey(R C | M) tag -> sets TD, TBX TBY TBZ TGX TGY
  local cfg tile block; cfg=$(grep BEST_PPCG_TUNED "$PT/raw/paper/ppcg_tune_$1_$2.txt" | sed 's/.*cfg=\[\(.*\)\]/\1/'); tile=${cfg%% |*}; block=${cfg##*| }
  TD=$PWD/bin/ppcgt_gen_$3; rm -rf $TD
  if [ $1 = 2d ]; then "$PT/ppcg_gen.sh" 2d ${2%x*} ${2#*x} $TD "$tile" "$block"; else "$PT/ppcg_gen.sh" 3d $2 - $TD "$tile" "$block"; fi
  cp $TD/kernel.hu $TD/matvec_ppcg_src_kernel.hu; cp $TD/kernel.hu $TD/cg_matvec_3d_ppcg_src_kernel.hu
  read TBX TBY TBZ TGX TGY < $TD/dims.txt; }
RES=$OUT/results.txt
[ -s "$RES" ] || printf "%-16s %-14s %5s %10s %6s %12s %12s\n" size method runs "avg ms" iters "max err" "min-max ms" > "$RES"

run10() {   # label bin tag
  local label=$1 bin=$2 tag=$3 times=() iters="" err="" t it e out
  for run in $(seq 1 10); do
    out=$("$bin"); echo "$out" > "$OUT/${tag}_run$run.txt"
    t=$(echo "$out" | grep -oP 'Time taken for main loop: \K[0-9.]+')
    it=$(echo "$out" | grep -oP 'Converged in \K[0-9]+'); e=$(echo "$out" | grep -oP 'Max abs error vs analytical solution: \K[0-9.eE+-]+')
    [ "$run" -gt 1 ] && times+=("$t"); iters=$it; err=$e
  done
  local avg mn mx
  avg=$(printf '%s\n' "${times[@]}" | LC_NUMERIC=C awk '{s+=$1} END {printf "%.4f", s/NR}')
  mn=$(printf '%s\n' "${times[@]}" | sort -g | head -1); mx=$(printf '%s\n' "${times[@]}" | sort -g | tail -1)
  printf "%-16s %-14s %5d %10s %6s %12s %12s\n" "$label" "$tag_m" 9 "$avg" "$iters" "$err" "$mn-$mx" | tee -a "$RES"
}

if [ "$WHICH" != 3d ]; then
  for s in $SIZES2D; do
    R=${s%x*}; C=${s#*x}; N=$((R*C)); D="$CG/tables/matvec/$N"; FC="$CG/tables/full_cg"
    read -r l1 l2 pc ca <<< "$(best_cfg 2d $s)"; read -r PBX PBY PGX PGY <<< "${PPCG2D[$s]}"; read -r bx by cy <<< "$(best_reg 2d $N)"
    nvcc -O3 -std=c++14 -arch=$ARCH "$FC/kernel_sparse_cg.cu" -o bin/cg_sparse_$N -DROWS=$R -DCOLS=$C 2>&1 | filt
    nvcc -O3 -std=c++14 -arch=$ARCH "$FC/kernel_mdh_cg.cu" "$CG/tables/matvec/cg_matvec_1.cu" -o bin/cg_mdhu_$N -DROWS=$R -DCOLS=$C $(flags2d $R $C 16 16 1 0) 2>&1 | filt
    nvcc -O3 -std=c++14 -arch=$ARCH "$FC/kernel_mdh_cg.cu" "$CG/tables/matvec/cg_matvec_1.cu" -o bin/cg_mdht_$N -DROWS=$R -DCOLS=$C $(flags2d $R $C $l1 $l2 $pc $ca) 2>&1 | filt
    nvcc -O3 -std=c++14 -arch=$ARCH "$FC/kernel_ppcg_cg.cu" "$D/matvec_ppcg_src_kernel.cu" -o bin/cg_ppcg_$N -DROWS=$R -DCOLS=$C -DPPCG_BLOCK_X=$PBX -DPPCG_BLOCK_Y=$PBY -DPPCG_GRID_X=$PGX -DPPCG_GRID_Y=$PGY -I"$D" 2>&1 | filt
    ppcgt_gen 2d $s 2d_$N
    nvcc -O3 -std=c++14 -arch=$ARCH "$FC/kernel_ppcg_cg.cu" $TD/kernel.cu -o bin/cg_ppcgt_$N -DROWS=$R -DCOLS=$C -DPPCG_BLOCK_X=$TBX -DPPCG_BLOCK_Y=$TBY -DPPCG_GRID_X=$TGX -DPPCG_GRID_Y=$TGY -I$TD 2>&1 | filt
    nvcc -O3 -std=c++14 -arch=$ARCH "$FC/kernel_cusparse_cg.cu" -o bin/cg_cusp_$N -DROWS=$R -DCOLS=$C -lcusparse 2>&1 | filt
    for mode in 0 1; do nvcc -O3 -std=c++14 -arch=$ARCH cg_handtuned.cu -o bin/cg_ht${mode}_$N -DDIM=2 -DROWS=$R -DCOLS=$C -DBX=$bx -DBY=$by -DCY=$cy -DMODE=$mode 2>&1 | filt; done
    check_gpu
    for m in sparse mdhu mdht ppcg ppcgt cusp ht0 ht1; do tag_m=$m; run10 "N=$N (${R}x${C})" ./bin/cg_${m}_$N "2d_${N}_$m"; done
  done
fi
if [ "$WHICH" != 2d ]; then
  for M in $SIZES3D; do
    N=$((M*M*M)); D="$CG/5_3d_extension/sizes/$M"; X="$CG/5_3d_extension"
    read -r l1 l2 l3 pc ca <<< "$(best_cfg 3d $M)"; read -r bx by ci <<< "$(best_reg 3d $M)"
    nvcc -O3 -std=c++14 -arch=$ARCH -DM=$M "$X/kernel_sparse_3d.cu" -o bin/cg_sparse_3d$M 2>&1 | filt
    nvcc -O3 -std=c++14 -arch=$ARCH "$X/kernel_mdh_3d.cu" "$X/cg_matvec_3d_1.cu" -o bin/cg_mdhu_3d$M $(flags3d $M 8 8 8 1 0) 2>&1 | filt
    nvcc -O3 -std=c++14 -arch=$ARCH "$X/kernel_mdh_3d.cu" "$X/cg_matvec_3d_1.cu" -o bin/cg_mdht_3d$M $(flags3d $M $l1 $l2 $l3 $pc $ca) 2>&1 | filt
    nvcc -O3 -std=c++14 -arch=$ARCH -DM=$M -I"$D" "$X/kernel_ppcg_3d.cu" "$D/cg_matvec_3d_ppcg_src_kernel.cu" -o bin/cg_ppcg_3d$M 2>&1 | filt
    ppcgt_gen 3d $M 3d_$M
    nvcc -O3 -std=c++14 -arch=$ARCH -DM=$M -DBX=$TBX -DBY=$TBY -DBZ=$TBZ -DGX=$TGX -DGY=$TGY -I$TD kernel_ppcg_3d_t.cu $TD/kernel.cu -o bin/cg_ppcgt_3d$M 2>&1 | filt
    nvcc -O3 -std=c++14 -arch=$ARCH -DM=$M "$X/kernel_cusparse_3d.cu" -o bin/cg_cusp_3d$M -lcusparse 2>&1 | filt
    for mode in 0 1; do nvcc -O3 -std=c++14 -arch=$ARCH cg_handtuned.cu -o bin/cg_ht${mode}_3d$M -DDIM=3 -DSIDE=$M -DBX=$bx -DBY=$by -DCY=$ci -DMODE=$mode 2>&1 | filt; done
    check_gpu
    for m in sparse mdhu mdht ppcg ppcgt cusp ht0 ht1; do tag_m=$m; run10 "N=$N (${M}^3)" ./bin/cg_${m}_3d$M "3d_${M}_$m"; done
  done
fi
echo "# methods: sparse=CSR, mdhu=MDH untuned, mdht=MDH tuned, ppcg=PPCG default, ppcgt=PPCG tuned (--sizes), cusp=cuSPARSE, ht0=hand-tuned (host-loop CG, same structure as paper), ht1=hand-tuned (fully GPU-resident CG)" >> "$RES"
