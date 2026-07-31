#!/bin/bash
# Proper tuning sweep for the MDH matrix-free stencil matvec kernel
# (cg_matvec_1.cu), never tuned before now -- always ran at the original
# untuned guess (L_CB=16x16, cache off). Sweeps tile size {8,16,32} for
# both L1/L2 dims (where the grid dimension divides evenly) x cache on/off.
set -e
cd "$(dirname "$0")"

declare -A ROWS=( [512]=16 [1024]=32 [2048]=32 [4096]=64 )
declare -A COLS=( [512]=32 [1024]=32 [2048]=64 [4096]=64 )
declare -A PBX=( [512]=16 [1024]=16 [2048]=16 [4096]=16 )
declare -A PBY=( [512]=16 [1024]=32 [2048]=32 [4096]=32 )
declare -A PGX=( [512]=1  [1024]=1  [2048]=2  [4096]=2 )
declare -A PGY=( [512]=1  [1024]=1  [2048]=1  [4096]=2 )

for size in 512 1024 2048 4096; do
  r=${ROWS[$size]}; c=${COLS[$size]}
  echo "############ N=$size (${r}x${c}) ############"
  best_ms=999999
  best_cfg=""
  ppcg_ms=""
  for l1 in 8 16 32; do
    for l2 in 8 16 32; do
      # tile must divide the grid dimension evenly
      if [ $((r % l1)) -ne 0 ] || [ $((c % l2)) -ne 0 ]; then continue; fi
      for cache in 0 1; do
        nwg1=$((r / l1)); nwg2=$((c / l2))
        nvcc -O3 bench_matvec.cu "$size/matvec_ppcg_src_kernel.cu" cg_matvec_1.cu -o /tmp/sweep_bin \
          -DROWS=$r -DCOLS=$c \
          -DTYPE_T=float -DTYPE_TS=float \
          -DCACHE_L_CB=$cache -DCACHE_P_CB=$cache \
          -DG_CB_RES_DEST_LEVEL=2 \
          -DG_CB_SIZE_L_1=$r -DG_CB_SIZE_L_2=$c \
          -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 \
          -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1 \
          -DNUM_WG_L_1=$nwg1 -DNUM_WG_L_2=$nwg2 \
          -DNUM_WI_L_1=$l1 -DNUM_WI_L_2=$l2 \
          -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0 \
          -DPPCG_BLOCK_X=${PBX[$size]} -DPPCG_BLOCK_Y=${PBY[$size]} -DPPCG_GRID_X=${PGX[$size]} -DPPCG_GRID_Y=${PGY[$size]} \
          -I"$size" -lcusparse 2>&1 | grep -viE "warning #68|warning #186|^generated" || true
        run_out=$(/tmp/sweep_bin)
        mdh_ms=$(echo "$run_out" | grep "^mdh" | awk '{print $2}')
        correct=$(echo "$run_out" | grep "^mdh" | awk '{print $4}')
        ppcg_ms=$(echo "$run_out" | grep "^ppcg" | awk '{print $2}')
        printf "  L_CB=%dx%d cache=%d -> %8s ms  (%s)   [ppcg: %s ms]\n" "$l1" "$l2" "$cache" "$mdh_ms" "$correct" "$ppcg_ms"
        is_best=$(LC_NUMERIC=C awk -v a="$mdh_ms" -v b="$best_ms" 'BEGIN{print (a<b)?1:0}')
        if [ "$is_best" = "1" ] && [ "$correct" = "yes" ]; then
          best_ms=$mdh_ms
          best_cfg="L_CB=${l1}x${l2} cache=$cache"
        fi
      done
    done
  done
  ratio=$(LC_NUMERIC=C awk -v p="$ppcg_ms" -v m="$best_ms" 'BEGIN{printf "%.3f", p/m}')
  echo "  >>> BEST: $best_cfg -> ${best_ms} ms  (ppcg/mdh ratio = ${ratio}x)"
  echo
done
rm -f /tmp/sweep_bin
