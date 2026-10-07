#!/bin/bash
# Tunes PPCG's --sizes (tile and thread-block sizes) per problem size, the PPCG counterpart of
# handtuned_baseline/mdh_tune_sweep.sh: stage 1 times every valid config (REPS=3), stage 2 re-times the top 3 with REPS=10
# (first dropped, 9 averaged). Compilation is done in parallel first; timing is serial on an otherwise idle GPU.
# Usage: ./ppcg_tune_sweep.sh 2d 4096 | 2d 16x32 | 3d 128     env: ARCH (sm_86), OUTDIR (raw), JOBS (6)
set -e
cd "$(dirname "$0")"
ARCH=${ARCH:-sm_86}; DIM=$1; S=$2; OUTDIR=${OUTDIR:-raw}; JOBS=${JOBS:-6}
mkdir -p "$OUTDIR"; OUT=$OUTDIR/ppcg_tune_${DIM}_${S}.txt; : > "$OUT"
if [ "$DIM" = 2d ] && [[ "$S" == *x* ]]; then R=${S%x*}; C=${S#*x}; elif [ "$DIM" = 2d ]; then R=$S; C=$S; fi
W=$(mktemp -d -p "$PWD"); trap 'rm -rf "$W"' EXIT
CFG=$W/cfgs.txt; : > $CFG
if [ "$DIM" = 2d ]; then
  for t0 in 8 16 32 64; do for t1 in 32 64 128; do
    for b0 in 1 2 4 8 16; do for b1 in 32 64 128; do
      [ $b0 -le $t0 ] && [ $b1 -le $t1 ] && [ $((b0*b1)) -le 1024 ] && [ $((b0*b1)) -ge 32 ] || continue
      echo "$t0,$t1 $b0,$b1" >> $CFG; done; done; done; done
else
  for t0 in 2 4 8 16; do for t1 in 4 8 16; do for t2 in 32 64; do
    for b0 in 1 2 4; do for b1 in 1 2 4 8; do for b2 in 16 32 64; do
      [ $b0 -le $t0 ] && [ $b1 -le $t1 ] && [ $b2 -le $t2 ] && [ $((b0*b1*b2)) -le 1024 ] && [ $((b0*b1*b2)) -ge 32 ] || continue
      echo "$t0,$t1,$t2 $b0,$b1,$b2" >> $CFG; done; done; done; done; done; done
fi
N=$(wc -l < $CFG); echo "# PPCG tuning ${DIM} ${S}: $N candidate configs (cfg = tile | block, PPCG --sizes index order)" | tee -a "$OUT"
build_one() {   # idx tile block
  local i=$1 tile=$2 block=$3 d=$W/c$1
  if [ "$DIM" = 2d ]; then ./ppcg_gen.sh 2d $R $C $d "$tile" "$block" || return 0
  else ./ppcg_gen.sh 3d $S - $d "$tile" "$block" || return 0; fi
  read bx by bz gx gy < $d/dims.txt
  if [ "$DIM" = 2d ]; then DEF="-DDIM=2 -DROWS=$R -DCOLS=$C"; else DEF="-DDIM=3 -DSIDE=$S"; fi
  nvcc -O3 -std=c++14 -arch=$ARCH bench_ppcg.cu $d/kernel.cu -I $d -o $d/b $DEF -DBX=$bx -DBY=$by -DBZ=$bz -DGX=$gx -DGY=$gy > $d/nvcc.log 2>&1 || rm -f $d/b
}
export -f build_one; export W R C S DIM ARCH
i=0; while read tile block; do i=$((i+1)); echo "$i $tile $block"; done < $CFG | xargs -P $JOBS -L1 bash -c 'build_one "$0" "$1" "$2"'
RANK=$W/rank.txt; : > $RANK
i=0; while read tile block; do i=$((i+1))
  [ -x $W/c$i/b ] || { echo "  stage1 cfg=[$tile | $block] skipped (generation/compile failed)" >> "$OUT"; continue; }
  line=$(REPS=3 $W/c$i/b 2>&1 | grep "^ppcg" || true)
  if echo "$line" | grep -q " ok$"; then ms=$(echo "$line" | awk '{print $2}'); echo "$ms $i $tile $block" >> $RANK; echo "  stage1 cfg=[$tile | $block] $ms ms" >> "$OUT"
  else echo "  stage1 cfg=[$tile | $block] skipped (incorrect or failed)" >> "$OUT"; fi
done < $CFG
echo "# stage 2: top 3 re-timed with REPS=10 (first dropped)" | tee -a "$OUT"
BEST=""; BCFG=""
while read ms i tile block; do
  line=$(REPS=10 $W/c$i/b 2>&1 | grep "^ppcg"); full=$(echo "$line" | awk '{print $2}')
  echo "  stage2 cfg=[$tile | $block] $full ms  ($line)" | tee -a "$OUT" > /dev/null
  if [ -z "$BEST" ] || [ "$(awk -v a=$full -v b=$BEST 'BEGIN{print (a<b)}')" = 1 ]; then BEST=$full; BCFG="$tile | $block"; fi
done < <(sort -g $RANK | head -3)
echo "BEST_PPCG_TUNED $BEST ms cfg=[$BCFG]" | tee -a "$OUT"
