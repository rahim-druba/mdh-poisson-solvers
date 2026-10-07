#!/bin/bash
# Tunes the MDH kernel (same search the paper's auto-tuner explores: tile sizes
# L_CB per dimension, per-thread coarsening P_CB on the slowest dim, L1/P cache
# on/off) at each size, so MDH and the hand-written kernel get comparable tuning
# effort. Stage 1: every valid config with REPS=3 (ranking). Stage 2: the top 3
# re-timed with the full protocol (REPS=10, first dropped). Only configs that
# compile and verify against the CPU reference are eligible.
# Usage: ./mdh_tune_sweep.sh 2d 4096 | ./mdh_tune_sweep.sh 3d 128
set -e
cd "$(dirname "$0")"
mkdir -p raw bin
CG="$(cd .. && pwd)"
ARCH=${ARCH:-sm_86}
DIM=$1; S=$2
# 2d also accepts a non-square grid as RxC (e.g. 16x32); R = slow (rows), C = contiguous (cols)
if [ "$DIM" = 2d ] && [[ "$S" == *x* ]]; then R=${S%x*}; C=${S#*x}; else R=$S; C=$S; fi
mkdir -p "${OUTDIR:-raw}"
OUT=${OUTDIR:-raw}/mdh_tune_${DIM}_${S}.txt
: > "$OUT"

build() {   # args: extra -D flags ; builds bin/mdh_sweep with REPS=$REPS
  if [ "$DIM" = 2d ]; then
    nvcc -O3 -std=c++14 -arch=$ARCH -DREPS=$REPS bench_handtuned.cu "$CG/tables/matvec/cg_matvec_1.cu" -o bin/mdh_sweep \
      -DDIM=2 -DROWS=$R -DCOLS=$C -DWITH_MDH -DMDH_ONLY -DMDH_KERNEL=cg_matvec_1 \
      -DTYPE_T=float -DTYPE_TS=float -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$R -DG_CB_SIZE_L_2=$C \
      -DL_CB_RES_DEST_LEVEL=1 -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_2=1 \
      -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0 "$@" 2>&1 | grep -E "error" || true
  else
    nvcc -O3 -std=c++14 -arch=$ARCH -DREPS=$REPS bench_handtuned.cu "$CG/5_3d_extension/cg_matvec_3d_1.cu" -o bin/mdh_sweep \
      -DDIM=3 -DSIDE=$S -DWITH_MDH -DMDH_ONLY -DMDH_KERNEL=cg_matvec_3d_1 \
      -DTYPE_T=float -DTYPE_TS=float -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=$S -DG_CB_SIZE_L_2=$S -DG_CB_SIZE_L_3=$S \
      -DL_CB_RES_DEST_LEVEL=1 -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_2=1 -DP_CB_SIZE_L_3=1 \
      -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_L_3=0 "$@" 2>&1 | grep -E "error" || true
  fi
}

flags_for() {   # l1 l2 [l3] pc cache -> echoes -D flags
  if [ "$DIM" = 2d ]; then
    local l1=$1 l2=$2 pc=$3 ca=$4
    echo "-DCACHE_L_CB=$ca -DCACHE_P_CB=$ca -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 -DP_CB_SIZE_L_1=$pc \
          -DNUM_WG_L_1=$((R/l1)) -DNUM_WG_L_2=$((C/l2)) -DNUM_WI_L_1=$((l1/pc)) -DNUM_WI_L_2=$l2"
  else
    local l1=$1 l2=$2 l3=$3 pc=$4 ca=$5
    echo "-DCACHE_L_CB=$ca -DCACHE_P_CB=$ca -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 -DL_CB_SIZE_L_3=$l3 -DP_CB_SIZE_L_1=$pc \
          -DNUM_WG_L_1=$((S/l1)) -DNUM_WG_L_2=$((S/l2)) -DNUM_WG_L_3=$((S/l3)) \
          -DNUM_WI_L_1=$((l1/pc)) -DNUM_WI_L_2=$l2 -DNUM_WI_L_3=$l3"
  fi
}

run_cfg() {   # echoes "<ms> <cfg-string>" if correct
  rm -f bin/mdh_sweep
  build "$@"
  [ -x bin/mdh_sweep ] || return 0
  local line; line=$(./bin/mdh_sweep 2>&1 | grep "MDH (generated)" || true)
  [ -n "$line" ] || return 0
  echo "$line" | grep -q " ok$" || return 0
  echo "$line" | awk '{print $3}'
}

# ---- stage 1: enumerate ----
CFGS=()
if [ "$DIM" = 2d ]; then
  for l1 in 4 8 16 32; do for l2 in 8 16 32 64; do
    [ $((R % l1)) -eq 0 ] && [ $((C % l2)) -eq 0 ] && [ $((l1*l2)) -le 1024 ] || continue
    for pc in 1 2; do [ $((l1 % pc)) -eq 0 ] || continue
      for ca in 0 1; do CFGS+=("$l1 $l2 $pc $ca"); done; done
  done; done
else
  for l1 in 2 3 4 6 8 12 16 24; do for l2 in 2 3 4 6 8 12 16 24; do for l3 in 4 8 12 16 24 32 64; do
    [ $((S % l1)) -eq 0 ] && [ $((S % l2)) -eq 0 ] && [ $((S % l3)) -eq 0 ] && [ $((l1*l2*l3)) -le 1024 ] || continue
    for pc in 1 2; do [ $((l1 % pc)) -eq 0 ] || continue
      for ca in 0 1; do CFGS+=("$l1 $l2 $l3 $pc $ca"); done; done
  done; done; done
fi
echo "# MDH tuning ${DIM} side ${S}: ${#CFGS[@]} candidate configs (cfg = L_CB dims, P_CB, cache)" | tee -a "$OUT"

REPS=3
RANK=$(mktemp)
for cfg in "${CFGS[@]}"; do
  ms=$(run_cfg $(flags_for $cfg))
  if [ -n "$ms" ]; then echo "$ms $cfg" >> "$RANK"; echo "  stage1 cfg=[$cfg] ${ms} ms" >> "$OUT"
  else echo "  stage1 cfg=[$cfg] skipped (compile error or incorrect)" >> "$OUT"; fi
done

# ---- stage 2: top 3 with the full protocol ----
REPS=10
echo "# stage 2: top 3 re-timed with REPS=10 (first dropped)" | tee -a "$OUT"
BEST_MS=""; BEST_CFG=""
while read -r ms cfg; do
  full=$(run_cfg $(flags_for $cfg))
  echo "  stage2 cfg=[$cfg] ${full} ms" | tee -a "$OUT"
  if [ -n "$full" ] && { [ -z "$BEST_MS" ] || [ "$(awk -v a="$full" -v b="$BEST_MS" 'BEGIN{print (a<b)}')" = 1 ]; }; then
    BEST_MS=$full; BEST_CFG=$cfg; fi
done < <(sort -g "$RANK" | head -3)
rm -f "$RANK"
echo "BEST_MDH_TUNED ${BEST_MS} ms cfg=[${BEST_CFG}]" | tee -a "$OUT"
