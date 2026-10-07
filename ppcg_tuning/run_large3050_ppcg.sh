#!/bin/bash
# Tuned PPCG at the large sizes on the RTX 3050 (for the GPU-resident CG comparison): generate candidates, then time them.
cd "$(dirname "$0")"; export ARCH=sm_86 JOBS=10; mkdir -p raw/large3050
for job in "2d 512 1" "2d 2048 1" "2d 4096 1" "3d 64 1" "3d 128 1" "3d 256 3"; do
  set -- $job
  echo "$1 $2 gen $(date +%T)" >> raw/large3050/status.txt
  ./ppcg_gen_cands.sh $1 $2 cands3050/$1_$2 $3 > /dev/null
  echo "$1 $2 time $(date +%T)" >> raw/large3050/status.txt
  ./ppcg_time_cands.sh $1 $2 cands3050/$1_$2 raw/large3050/ppcg_$1_$2.txt > /dev/null 2>&1
  echo "$1 $2 done $(date +%T)" >> raw/large3050/status.txt
done
echo ALL_DONE >> raw/large3050/status.txt
