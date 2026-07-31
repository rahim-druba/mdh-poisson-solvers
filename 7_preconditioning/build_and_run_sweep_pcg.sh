#!/bin/bash
# Builds and runs plain CG (cuSPARSE), Jacobi-PCG, and ILU(0)-PCG at 4
# square grid sizes (16x16/32x32/64x64/128x128 -> N=256/1024/4096/16384),
# same 10-run/9-averaged protocol as ../tables/full_cg/build_and_run_sweep.sh.
# Square grids only (not the 512/2048 non-square sizes used elsewhere) --
# preconditioning doesn't depend on squareness, and clean squares keep
# this sweep's FULLN convention identical to 1_sparse_rewrite/kernel_sparse.cu.
set -e
cd "$(dirname "$0")"

for FULLN in 18 34 66 130; do
  nvcc -O3 -DFULLN=$FULLN kernel_plaincg_baseline.cu -o "plaincg_$FULLN" -lcusparse 2>&1 | grep -v "^$" || true
  nvcc -O3 -DFULLN=$FULLN kernel_jacobi_pcg.cu -o "jacobi_$FULLN" -lcusparse 2>&1 | grep -v "^$" || true
  nvcc -O3 -DFULLN=$FULLN kernel_ilu0_pcg.cu -o "ilu0_$FULLN" -lcusparse 2>&1 | grep -v "^$" || true
done

echo "All 12 binaries built."
echo

printf "%-10s %-10s %8s %10s %14s %14s\n" "size" "method" "runs" "avg ms" "iterations" "max err"
printf -- "----------------------------------------------------------------------------\n"

for FULLN in 18 34 66 130; do
  M=$((FULLN - 2))
  N=$((M * M))
  for method in plaincg jacobi ilu0; do
    bin="./${method}_${FULLN}"
    times=()
    iters=""; err=""
    for run in $(seq 1 10); do
      out=$("$bin")
      t=$(echo "$out" | grep -oP 'Time taken for main loop: \K[0-9.]+')
      it=$(echo "$out" | grep -oP 'Converged in \K[0-9]+')
      e=$(echo "$out" | grep -oP 'Max abs error vs analytical solution: \K[0-9.eE+-]+')
      if [ "$run" -gt 1 ]; then times+=("$t"); fi
      iters="$it"; err="$e"
    done
    avg=$(printf '%s\n' "${times[@]}" | LC_NUMERIC=C awk '{s+=$1} END {printf "%.4f", s/NR}')
    printf "%-10s %-10s %8d %10s %14s %14s\n" "N=$N" "$method" "${#times[@]}" "$avg" "$iters" "$err"
  done
done
