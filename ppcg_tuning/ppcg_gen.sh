#!/bin/bash
# Generates a PPCG CUDA kernel for the Poisson stencil matvec (same C source as the paper's, only the sizes differ).
# Usage: ./ppcg_gen.sh 2d R C OUTDIR ["tile0,tile1" "block0,block1"]     (block order = PPCG index order; empty = PPCG default)
#        ./ppcg_gen.sh 3d M - OUTDIR ["tile0,tile1,tile2" "block0,block1,block2"]
# Writes OUTDIR/kernel.cu, kernel.hu, dims.txt ("BX BY BZ GX GY" as launched by PPCG's own host code).
set -e
cd "$(dirname "$0")"
PPCG=${PPCG:-ppcg}   # PPCG 0.08.3 binary; set PPCG=/path/to/ppcg if it is not on the PATH
DIM=$1; A=$2; B=$3; OUT=$4; TILE=$5; BLOCK=$6
mkdir -p "$OUT"; W=$(mktemp -d)
if [ "$DIM" = 2d ]; then sed "s/#define R 64/#define R $A/;s/#define C 128/#define C $B/" template2d.c > $W/src.c
else sed "s/#define M 32/#define M $A/" template3d.c > $W/src.c; fi
SZ=""
[ -n "$TILE" ] && SZ="{ kernel[i] -> tile[$TILE]; kernel[i] -> block[$BLOCK] }"
(cd $W && if [ -n "$SZ" ]; then "$PPCG" --target=cuda --sizes="$SZ" src.c; else "$PPCG" --target=cuda src.c; fi) > $W/ppcg.log 2>&1 || { cat $W/ppcg.log; exit 1; }
sed 's/src_kernel.hu/kernel.hu/' $W/src_kernel.cu > "$OUT/kernel.cu"; cp $W/src_kernel.hu "$OUT/kernel.hu"
bl=$(grep -oP 'k0_dimBlock\(\K[^)]*' $W/src_host.cu | tr -d ' ' | tr ',' ' ')
gr=$(grep -oP 'k0_dimGrid\(\K[^)]*' $W/src_host.cu | tr -d ' ' | tr ',' ' ')
set -- $bl; bx=$1; by=${2:-1}; bz=${3:-1}; set -- $gr; gx=$1; gy=${2:-1}
echo "$bx $by $bz $gx $gy" > "$OUT/dims.txt"; rm -rf $W
