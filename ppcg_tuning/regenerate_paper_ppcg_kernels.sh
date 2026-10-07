#!/bin/bash
# Regenerates the PPCG kernels with PPCG's DEFAULT schedule for the sizes of the paper (default PPCG columns):
#   tables/matvec/<N>/matvec_ppcg_src_kernel.cu (2D, N = 512 1024 2048 4096 [8192])  and
#   5_3d_extension/sizes/<M>/cg_matvec_3d_ppcg_src_kernel.cu (3D, M = 8 16 24 32)
# from the plain C sources in the same folders. Needs PPCG 0.08.3 (set PPCG=/path/to/ppcg if it is not on the PATH).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PPCG=${PPCG:-ppcg}
for n in 512 1024 2048 4096 8192; do [ -f "$ROOT/tables/matvec/$n/matvec_ppcg_src.c" ] && (cd "$ROOT/tables/matvec/$n" && "$PPCG" --target=cuda matvec_ppcg_src.c > /dev/null); done
for m in 8 16 24 32; do [ -f "$ROOT/5_3d_extension/sizes/$m/cg_matvec_3d_ppcg_src.c" ] && (cd "$ROOT/5_3d_extension/sizes/$m" && "$PPCG" --target=cuda cg_matvec_3d_ppcg_src.c > /dev/null); done
echo "done"
