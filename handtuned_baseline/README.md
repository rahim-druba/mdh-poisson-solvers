# Experiments added in the revision: second GPU, hand-tuned baseline, tuned PPCG, fully generated CG

These folders (`handtuned_baseline/`, `ppcg_tuning/`, `mdh_specs/`, plus `roofline/` additions) hold everything behind the
revised comparison of MDH-generated kernels with a hand-tuned matrix-free kernel, PPCG with tuned tile/block sizes, CSR and cuSPARSE,
on two NVIDIA GPUs. Scripts expect the folder layout of this repository (`tables/` and `5_3d_extension/` next to this folder).

## Hardware and software
| | RTX 3050 Laptop | RTX 5090 |
|---|---|---|
| compute capability / memory / peak bandwidth | 8.6 / 4 GB / 192 GB/s | 12.0 / 32 GB / about 1.79 TB/s |
| CUDA toolkit | 11.7 (nvcc 11.7.99), GCC 9.4.0, driver 535.230.02, Ubuntu 20.04 | 12.9, GCC 13.3, Ubuntu 24.04 |
| `-arch` flag of all runs here | `sm_86` | `sm_120` |

PPCG 0.08.3 (clang 10) generated the PPCG kernels; the MDH kernels come from the generator in
https://github.com/rahim-druba/mdh-cuda-stencil-kernels (tested with commit `8dae1df`), which extends the MDH PACT 2019 artifact (see the root README).
Note: the original tables of `tables/` and `5_3d_extension/` were built without an `-arch` flag (nvcc 11.7 default `sm_52`, PTX compiled by the
driver); all experiments in this folder use the exact architecture for every method.

## Quick start (from the repository root)
```bash
git clone https://github.com/rahim-druba/mdh-cuda-stencil-kernels && export MDH_FRAMEWORK=$PWD/mdh-cuda-stencil-kernels
mdh_specs/generate.sh                           # MDH kernels: stencils (2D, 3D), dot product, axpy
ppcg_tuning/regenerate_paper_ppcg_kernels.sh    # PPCG default-schedule kernels at the paper sizes (needs ppcg on the PATH)
cd handtuned_baseline
ARCH=sm_86 ./paper_sizes.sh run both            # paper sizes, one session: CSR, cuSPARSE, PPCG default/tuned, MDH untuned/tuned, hand-tuned
ARCH=sm_86 ./paper_cg.sh both                   # full CG solves at the paper sizes (host-loop solver), all variants
```
`paper_sizes.sh run` reads the tuned configurations from the shipped logs (`raw/paper_sizes/mdh_tune_*.txt`, `../ppcg_tuning/raw/paper/ppcg_tune_*.txt`);
run `./paper_sizes.sh tune both` and `../ppcg_tuning/run_paper_ppcg_tune.sh` first to re-tune on another GPU.

## Which script produces which result
| Result in the paper | Script | Logs |
|---|---|---|
| Matrix-vector product at the paper sizes (3050), tuned and untuned | `paper_sizes.sh`, `mdh_tune_sweep.sh`, `bench_handtuned.cu` | `raw/paper_sizes/`, `raw/mdh_tune_*.txt`, `raw/2d_*.txt`, `raw/3d_*.txt` |
| PPCG with tuned tile/block sizes | `../ppcg_tuning/ppcg_tune_sweep.sh`, `run_paper_ppcg_tune.sh` | `../ppcg_tuning/raw/paper/` |
| RTX 5090, large sizes: CSR, cuSPARSE, hand-tuned, MDH (3 sessions) | `run_5090_formats.sh`, `bench_csr.cu` | `raw_5090/formats*`, `raw_5090/mdh_tune_*.txt` |
| RTX 5090, PPCG default and tuned | `../ppcg_tuning/ppcg_gen_cands.sh` (on a machine with PPCG), `ppcg_time_cands.sh` (on the GPU machine) | `../ppcg_tuning/raw_5090/` |
| Full CG solve, host-loop solver, all variants | `paper_cg.sh`, `cg_handtuned.cu` | `raw/paper_cg/` (`results.txt`; per-run logs in `runs.tar.gz`) |
| GPU-resident CG, five matrix-vector products | `run_cg_resident2.sh`, `cg_resident.cu`, `parse_cgres2.py` | `raw/cg_resident2_3050/`, `raw_5090/cg_resident2/` |
| Fully MDH-generated CG iteration | `run_cg_full.sh`, `cg_resident.cu` (`VEC=1`), `../mdh_specs/{dot,axpy}.cpp`, `parse_cgfull.py` | `raw/cg_full_3050/`, `raw_5090/cg_full/` |
| Regime map (effective bandwidth) and cross-GPU transfer | `derived_tables.py` (no GPU needed) | computed from the logs above |
| Nsight Compute re-profile of the tuned MDH and hand-tuned kernels | `../roofline/profile_roofline_3d_tuned_final.sh` (needs root) | `../roofline/*_tuned_ncu.txt`, `ht_3d_*_ncu.txt` |

`results/` holds the markdown tables built by `build_tables.py`.

## Measurement protocol and validity checks
- Isolated kernels: at least 300 ms warm-up, 200 timed launches x 10 repetitions, first dropped, nine averaged (`cudaEvent`); every kernel is checked against a CPU reference.
- Full CG solves: 10 runs per solver, first dropped, nine averaged; absolute residual tolerance 1e-6, at most 5000 iterations, fp32.
- GPU-resident CG at large sizes: fixed 100 iterations (fp32 CG does not reach 1e-6 there), six repetitions, first dropped; the final residual norms of all variants are compared.
- Before every timed session: persistence mode on, AC power, no throttling reasons reported by `nvidia-smi -q -d PERFORMANCE`, no other GPU process, no browser open. `paper_sizes.sh` and `paper_cg.sh` abort if throttling or battery power is detected; the other scripts record the GPU clock, power and temperature before each size (`gpu_before_*.txt`).
- `raw*/_not_used/` contains sessions that were discarded or superseded (battery power, browser open, script bug, earlier versions) and why.

## Layout
`bench_*.cu`, `cg_*.cu`: sources; `*.sh`: drivers; `cfg_cgres_*.txt`: hand-tuned kernel configuration per size; `mdh_vec/`: generated dot-product and axpy kernels
(not tracked, written by `mdh_specs/generate.sh`); `bin/`: compiled binaries (not tracked).

## Limits
Two NVIDIA GPUs only (CUDA backend); no other vendor was available. Tuned MDH configurations come from an exhaustive sweep around the generated kernel, not from a run of ATF.
