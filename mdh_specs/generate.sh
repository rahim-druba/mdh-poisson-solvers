#!/bin/bash
# Generates the MDH kernels used by the experiments of this repository from the four specifications:
#   cg_matvec (2D 5-point stencil, ../2_mdh_sparse/.../cg_matvec.cpp), cg_matvec_3d (3D 7-point stencil),
#   dot (dot product, a reduction), axpy (vector update, a map with a scalar input).
# Needs the MDH generator framework: https://github.com/rahim-druba/mdh-cuda-stencil-kernels (tested with commit 8dae1df),
# C++14 compiler, CMake. Usage:   MDH_FRAMEWORK=/path/to/mdh-cuda-stencil-kernels ./generate.sh
# Output (not tracked by git, see .gitignore): tables/matvec/cg_matvec_1.cu, 5_3d_extension/cg_matvec_3d_1.cu,
#                                              handtuned_baseline/mdh_vec/{dot_1.cu,axpy_1.cu}
set -e
cd "$(dirname "$0")"
F=${MDH_FRAMEWORK:?set MDH_FRAMEWORK to the checkout of mdh-cuda-stencil-kernels}
if [ ! -f "$F/build/libmdh_cuda_generator.a" ]; then
  mkdir -p "$F/build" && (cd "$F/build" && cmake .. > /dev/null && make mdh_cuda_generator)
fi
mkdir -p build_specs && cd build_specs
for spec in ../../2_mdh_sparse/mdh_generator_source/spec/cg_matvec.cpp ../cg_matvec_3d.cpp ../dot.cpp ../axpy.cpp; do
  name=$(basename "$spec" .cpp)
  g++ -std=c++14 -w -I"$F/framework" -I"$F/framework/include" "$spec" "$F/build/libmdh_cuda_generator.a" -o "gen_$name"
  "./gen_$name"
done
cp cg_matvec_1.cu ../../tables/matvec/
cp cg_matvec_3d_1.cu ../../5_3d_extension/
mkdir -p ../../handtuned_baseline/mdh_vec && cp dot_1.cu axpy_1.cu ../../handtuned_baseline/mdh_vec/
echo "generated: tables/matvec/cg_matvec_1.cu 5_3d_extension/cg_matvec_3d_1.cu handtuned_baseline/mdh_vec/{dot_1.cu,axpy_1.cu}"
