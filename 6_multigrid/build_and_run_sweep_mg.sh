#!/bin/bash
# Builds and runs the geometric multigrid solver at 4 sizes (M = 2^k - 1:
# 31/63/127/255 -> N=961/3969/16129/65025), same 10-run/9-averaged protocol
# as ../tables/full_cg/build_and_run_sweep.sh.
set -e
cd "$(dirname "$0")"

for M in 31 63 127 255; do
  nvcc -O3 -DM=$M kernel_multigrid_2d.cu -o "mg_$M" 2>&1 | grep -v "^$" || true
done

echo "All 4 binaries built."
echo

printf "%-8s %8s %10s %14s %14s\n" "size" "runs" "avg ms" "V-cycles" "max err"
printf -- "----------------------------------------------------------------\n"

for M in 31 63 127 255; do
  N=$((M * M))
  bin="./mg_$M"
  times=()
  cycles=""; err=""
  for run in $(seq 1 10); do
    out=$("$bin")
    t=$(echo "$out" | grep -oP 'Time taken for main loop: \K[0-9.]+')
    c=$(echo "$out" | grep -oP 'Converged in \K[0-9]+')
    e=$(echo "$out" | grep -oP 'Max abs error vs analytical solution: \K[0-9.eE+-]+')
    if [ "$run" -gt 1 ]; then times+=("$t"); fi
    cycles="$c"; err="$e"
  done
  avg=$(printf '%s\n' "${times[@]}" | LC_NUMERIC=C awk '{s+=$1} END {printf "%.4f", s/NR}')
  printf "%-8s %8d %10s %14s %14s\n" "N=$N (${M}^2)" "${#times[@]}" "$avg" "$cycles" "$err"
done
