#!/bin/bash
# PPCG --sizes tuning sweeps at the paper's sizes (RTX 3050). Output: raw/paper/ppcg_tune_*.txt
cd "$(dirname "$0")"
export OUTDIR=raw/paper JOBS=${JOBS:-10}
for s in 16x32 32x32 32x64 64x64; do ./ppcg_tune_sweep.sh 2d $s; done
for m in 8 16 24 32; do ./ppcg_tune_sweep.sh 3d $m; done
echo ALL_DONE
