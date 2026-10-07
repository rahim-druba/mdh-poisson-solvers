# Measured DRAM bandwidth, 3D matrix-vector product (RTX 3050 Laptop)

DRAM throughput of one steady-state matvec launch per size (Nsight Compute, metric `dram__bytes.sum.per_second`), and the same value as a percentage of the 192 GB/s
theoretical peak (128-bit bus, 6001 MHz memory clock). Nsight's own "percent of peak sustained" corresponds to about 171 GB/s and is not used here.
CSR, cuSPARSE, MDH untuned and PPCG default come from `results_3d.md`; MDH tuned and hand-tuned come from `profile_roofline_3d_tuned_final.sh`.

| kernel | N=512 | N=4096 | N=13824 | N=32768 |
|---|---|---|---|---|
| CSR | 6.61 GB/s (3.4%) | 48.52 GB/s (25.3%) | 96.36 GB/s (50.2%) | 113.55 GB/s (59.1%) |
| cuSPARSE | 4.51 GB/s (2.3%) | 35.36 GB/s (18.4%) | 77.18 GB/s (40.2%) | 98.52 GB/s (51.3%) |
| MDH untuned | 0.71 GB/s (0.4%) | 5.11 GB/s (2.7%) | 15.10 GB/s (7.9%) | 24.26 GB/s (12.6%) |
| PPCG default | 1.02 GB/s (0.5%) | 2.63 GB/s (1.4%) | 3.98 GB/s (2.1%) | 4.30 GB/s (2.2%) |
| MDH tuned | 0.78 GB/s (0.4%) | 5.73 GB/s (3.0%) | 16.50 GB/s (8.6%) | 34.37 GB/s (17.9%) |
| hand-tuned | 0.80 GB/s (0.4%) | 5.59 GB/s (2.9%) | 17.49 GB/s (9.1%) | 28.87 GB/s (15.0%) |

The SM throughput of the same launches (MDH tuned 0.8, 6.0, 16.7, 31.0%; hand-tuned 2.9, 10.2, 20.3, 20.1%) is in `mdh_3d_*_tuned_ncu.txt` and `ht_3d_*_ncu.txt`.
