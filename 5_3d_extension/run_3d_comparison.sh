#!/bin/bash
# Runs all four 3D full-CG solvers, 10 runs each (first dropped as warmup,
# remaining 9 averaged) -- same protocol as ../tables/full_cg.
set -e
cd "$(dirname "$0")"

printf "%-10s %8s %10s %14s %14s\n" "method" "runs" "avg ms" "iterations" "max err"
printf -- "--------------------------------------------------------------\n"

for method in sparse mdh ppcg cusparse; do
    bin="./kernel_${method}_3d"
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
    printf "%-10s %8d %10s %14s %14s\n" "$method" "${#times[@]}" "$avg" "$iters" "$err"
done
