#!/bin/bash
# Full pipeline: MDH tuning sweeps, then hand-tuned baseline (REPS=10), all sizes.
cd "$(dirname "$0")"
for S in 64 512 2048 4096; do ./mdh_tune_sweep.sh 2d $S; done
for M in 16 32 64 128 256; do ./mdh_tune_sweep.sh 3d $M; done
./build_and_run.sh
