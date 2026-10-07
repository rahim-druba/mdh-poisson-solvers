#!/bin/bash
# GPU-resident CG at large sizes: matrix-free (hand-tuned) vs hand-written CSR vs cuSPARSE, fixed iterations (see cg_resident.cu).
# Usage: ARCH=sm_86 CFG=cfg_cgres_3050.txt OUT=raw/cg_resident_3050 ./run_cg_resident.sh     (laptop: Chrome closed, charger on)
#        on the RTX 5090 machine with ARCH=sm_120 CFG=cfg_cgres_5090.txt OUT=raw_5090/cg_resident (after setting up CUDA 12.9: CUDA_HOME, CPATH, LIBRARY_PATH, CUDA_VISIBLE_DEVICES)
set -e
cd "$(dirname "$0")"
ARCH=${ARCH:-sm_86}; CFG=${CFG:?}; OUT=${OUT:?}; mkdir -p "$OUT" bin; : > "$OUT/status.txt"
while read dim s bx by cy; do
  echo "$dim $s started $(date +%T)" >> "$OUT/status.txt"
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader > "$OUT/gpu_before_${dim}_$s.txt"
  if [ $dim = 2d ]; then DEF="-DDIM=2 -DROWS=$s -DCOLS=$s"; else DEF="-DDIM=3 -DSIDE=$s"; fi
  : > "$OUT/cgres_${dim}_$s.txt"
  for mv in 0 1 2; do
    nvcc -O3 -std=c++14 -arch=$ARCH cg_resident.cu -o bin/cgres_${dim}_${s}_$mv $DEF -DBX=$bx -DBY=$by -DCY=$cy -DMV=$mv -lcusparse 2>&1 | grep -iE "error" || true
    ./bin/cgres_${dim}_${s}_$mv | tee -a "$OUT/cgres_${dim}_$s.txt"
  done
  echo "$dim $s done $(date +%T)" >> "$OUT/status.txt"
done < "$CFG"
echo ALL_DONE >> "$OUT/status.txt"
