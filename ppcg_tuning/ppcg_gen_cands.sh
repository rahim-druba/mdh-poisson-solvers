#!/bin/bash
# LOCAL half of the PPCG sweep for machines without PPCG/clang (the RTX 5090 server): generates every candidate kernel
# (same candidate lists as ppcg_tune_sweep.sh, plus PPCG's own default schedule as candidate 0) into OUTDIR/c<i>/.
# Usage: ./ppcg_gen_cands.sh 2d 8192 OUTDIR [STRIDE]   (STRIDE>1 keeps every STRIDE-th candidate, always keeps the default)
set -e
cd "$(dirname "$0")"
DIM=$1; S=$2; OUT=$(realpath -m "$3"); STRIDE=${4:-1}; JOBS=${JOBS:-10}
rm -rf "$OUT"; mkdir -p "$OUT"; ALL=$OUT/all.txt; CFG=$OUT/cfgs.txt; : > $ALL
if [ "$DIM" = 2d ]; then
  for t0 in 8 16 32 64; do for t1 in 32 64 128; do for b0 in 1 2 4 8 16; do for b1 in 32 64 128; do
    [ $b0 -le $t0 ] && [ $b1 -le $t1 ] && [ $((b0*b1)) -le 1024 ] && [ $((b0*b1)) -ge 32 ] || continue
    echo "$t0,$t1 $b0,$b1" >> $ALL; done; done; done; done
else
  for t0 in 2 4 8 16; do for t1 in 4 8 16; do for t2 in 32 64; do for b0 in 1 2 4; do for b1 in 1 2 4 8; do for b2 in 16 32 64; do
    [ $b0 -le $t0 ] && [ $b1 -le $t1 ] && [ $b2 -le $t2 ] && [ $((b0*b1*b2)) -le 1024 ] && [ $((b0*b1*b2)) -ge 32 ] || continue
    echo "$t0,$t1,$t2 $b0,$b1,$b2" >> $ALL; done; done; done; done; done; done
fi
echo "0 default default" > $CFG
awk -v s=$STRIDE 'NR%s==1 || s==1 {print NR" "$0}' $ALL >> $CFG
gen_one() { local i=$1 tile=$2 block=$3 d=$4/c$1
  if [ "$tile" = default ]; then tile=""; block=""; fi
  if [ "$DIM" = 2d ]; then ./ppcg_gen.sh 2d $S $S $d "$tile" "$block" || echo FAIL > $d.fail
  else ./ppcg_gen.sh 3d $S - $d "$tile" "$block" || echo FAIL > $d.fail; fi; }
export -f gen_one; export DIM S
xargs -P $JOBS -L1 bash -c 'gen_one "$0" "$1" "$2" '"$OUT" < $CFG
echo "generated $(wc -l < $CFG) candidates in $OUT ($(ls $OUT/*.fail 2>/dev/null | wc -l) failed)"
