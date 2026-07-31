#!/bin/bash
# Real tuning sweep for the MDH GEMM kernel: 3 row-tile x 2 col-tile x 3
# reduction-tile sizes = 18 configs, fixed at 16x16 threads/block (256,
# a safe/standard block shape). Reports best config found vs PPCG.
cd "$(dirname "$0")"

SIZE=$1
PBX=$2; PBY=$3; PGX=$4; PGY=$5

best_ms=999999
best_cfg=""
ppcg_ms=""

for l1 in 64 128 256; do
  for l2 in 64 128; do
    for r1 in 8 16 32; do
      pcb1=$((l1/16)); pcb2=$((l2/16))
      nwg1=$((SIZE/l1)); nwg2=$((SIZE/l2))
      if [ $((SIZE % l1)) -ne 0 ] || [ $((SIZE % l2)) -ne 0 ] || [ $((SIZE % r1)) -ne 0 ]; then continue; fi
      out=$(nvcc -O3 bench_matmul.cu "$SIZE/matmul_ppcg_src_kernel.cu" gemm_1.cu -o /tmp/gemm_sweep_bin \
        -DGEMM_N=$SIZE -DTYPE_T=float -DTYPE_TS=float \
        -DCACHE_L_CB=1 -DCACHE_P_CB=1 \
        -DG_CB_RES_DEST_LEVEL=2 -DL_CB_RES_DEST_LEVEL=1 -DP_CB_RES_DEST_LEVEL=0 \
        -DG_CB_SIZE_L_1=$SIZE -DG_CB_SIZE_L_2=$SIZE -DG_CB_SIZE_R_1=$SIZE \
        -DL_CB_SIZE_L_1=$l1 -DL_CB_SIZE_L_2=$l2 -DL_CB_SIZE_R_1=$r1 \
        -DP_CB_SIZE_L_1=$pcb1 -DP_CB_SIZE_L_2=$pcb2 -DP_CB_SIZE_R_1=1 \
        -DNUM_WG_L_1=$nwg1 -DNUM_WG_L_2=$nwg2 -DNUM_WG_R_1=1 \
        -DNUM_WI_L_1=16 -DNUM_WI_L_2=16 -DNUM_WI_R_1=1 \
        -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_R_1=0 \
        -DPPCG_BLOCK_X=$PBX -DPPCG_BLOCK_Y=$PBY -DPPCG_GRID_X=$PGX -DPPCG_GRID_Y=$PGY \
        -I"$SIZE" -lcublas 2>&1)
      if echo "$out" | grep -qi "error"; then
        printf "  L_CB=%dx%dx%d -> BUILD FAILED (likely shared-mem/register limit)\n" "$l1" "$l2" "$r1"
        continue
      fi
      run_out=$(/tmp/gemm_sweep_bin 2>&1)
      mdh_ms=$(echo "$run_out" | grep "^mdh" | awk '{print $2}')
      correct=$(echo "$run_out" | grep "^mdh" | awk '{print $5}')
      ppcg_ms=$(echo "$run_out" | grep "^ppcg" | awk '{print $2}')
      printf "  L_CB=%dx%dx%d -> %9s ms  (%s)   [ppcg: %s ms]\n" "$l1" "$l2" "$r1" "$mdh_ms" "$correct" "$ppcg_ms"
      if [ "$correct" = "yes" ]; then
        is_best=$(LC_NUMERIC=C awk -v a="$mdh_ms" -v b="$best_ms" 'BEGIN{print (a<b)?1:0}')
        if [ "$is_best" = "1" ]; then
          best_ms=$mdh_ms
          best_cfg="L_CB=${l1}x${l2}x${r1}"
        fi
      fi
    done
  done
done
ratio=$(LC_NUMERIC=C awk -v p="$ppcg_ms" -v m="$best_ms" 'BEGIN{printf "%.3f", p/m}')
echo "  >>> BEST N=$SIZE: $best_cfg -> ${best_ms} ms  (ppcg/mdh ratio = ${ratio}x, last ppcg=${ppcg_ms}ms)"
rm -f /tmp/gemm_sweep_bin
