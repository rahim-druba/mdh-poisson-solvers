#!/bin/bash
# SERVER half: compiles and times the pre-generated candidates (from ppcg_gen_cands.sh) on the local GPU, same protocol as
# ppcg_tune_sweep.sh (stage 1 REPS=3 all, stage 2 top 3 REPS=10). Candidate 0 = PPCG default, timed with REPS=10 separately.
# Usage: ./ppcg_time_cands.sh 2d 8192 CANDDIR OUTFILE      env: ARCH, JOBS
set -e
cd "$(dirname "$0")"
ARCH=${ARCH:-sm_120}; DIM=$1; S=$2; D=$3; OUT=$4; JOBS=${JOBS:-16}
if [ "$DIM" = 2d ]; then DEF="-DDIM=2 -DROWS=$S -DCOLS=$S"; else DEF="-DDIM=3 -DSIDE=$S"; fi
build_one() { local c=$2/c$1; [ -f $c/dims.txt ] || return 0; read bx by bz gx gy < $c/dims.txt
  nvcc -O3 -std=c++14 -arch=$ARCH bench_ppcg.cu $c/kernel.cu -I $c -o $c/b $3 -DBX=$bx -DBY=$by -DBZ=$bz -DGX=$gx -DGY=$gy > $c/nvcc.log 2>&1 || rm -f $c/b; }
export -f build_one; export ARCH
awk '{print $1}' $D/cfgs.txt | xargs -P $JOBS -I{} bash -c "build_one {} $D '$DEF'"
: > "$OUT"; echo "# PPCG ${DIM} ${S} on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d '\n') (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >> "$OUT"
RANK=$D/rank.txt; : > $RANK
while read i tile block; do
  [ "$i" = 0 ] && continue
  [ -x $D/c$i/b ] || { echo "  stage1 cfg=[$tile | $block] skipped (generation/compile failed)" >> "$OUT"; continue; }
  line=$(REPS=3 $D/c$i/b 2>&1 | grep "^ppcg" || true)
  if echo "$line" | grep -q " ok$"; then ms=$(echo "$line" | awk '{print $2}'); echo "$ms $i $tile $block" >> $RANK; echo "  stage1 cfg=[$tile | $block] $ms ms" >> "$OUT"
  else echo "  stage1 cfg=[$tile | $block] skipped (incorrect or failed)" >> "$OUT"; fi
done < $D/cfgs.txt
echo "# stage 2: top 3 re-timed with REPS=10 (first dropped)" >> "$OUT"
BEST=""; BCFG=""
while read ms i tile block; do
  line=$(REPS=10 $D/c$i/b 2>&1 | grep "^ppcg"); full=$(echo "$line" | awk '{print $2}')
  echo "  stage2 cfg=[$tile | $block] $full ms  ($line)" >> "$OUT"
  if [ -z "$BEST" ] || [ "$(awk -v a=$full -v b=$BEST 'BEGIN{print (a<b)}')" = 1 ]; then BEST=$full; BCFG="$tile | $block"; fi
done < <(sort -g $RANK | head -3)
echo "BEST_PPCG_TUNED $BEST ms cfg=[$BCFG]" >> "$OUT"
if [ -x $D/c0/b ]; then echo "PPCG_DEFAULT $(REPS=10 $D/c0/b 2>&1 | grep '^ppcg')" >> "$OUT"; else echo "PPCG_DEFAULT failed to build" >> "$OUT"; fi
tail -2 "$OUT"
