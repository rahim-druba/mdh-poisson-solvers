#!/usr/bin/env python3
"""Reproduces two tables of the paper that are derived from the raw logs (no GPU needed):
  1. regime map on the RTX 5090: effective bandwidth = minimum bytes / measured time, from raw_5090/formats (session 1)
     matrix-free 8 B per point; CSR 52 B per row (2D) or 68 B per row (3D); peak 1792 GB/s
  2. cross-GPU transfer of tuned MDH configurations: run time of the other GPU's best configuration relative to the GPU's own best,
     from the stage-1 lines of raw/mdh_tune_* (RTX 3050) and raw_5090/mdh_tune_* (RTX 5090)
Usage: python3 derived_tables.py"""
import re, os
H = os.path.dirname(os.path.abspath(__file__))
def rd(p):
    return open(p).read() if os.path.exists(p) else ''
def tm(txt, pat):
    m = re.search(pat + r'\s+([0-9.]+) ms', txt); return float(m.group(1)) if m else None
def ceil(t):
    m = re.search(r'HAND-TUNED CEILING:\s+.+?\s+([0-9.]+) ms', t); return float(m.group(1)) if m else None
print('== 1. RTX 5090 effective bandwidth (% of 1792 GB/s) and speedups')
print('%-9s %8s %8s %8s %8s | CSR/HT bytes-ratio' % ('size', 'CSR', 'cuSPARSE', 'HT', 'MDHt'))
F = os.path.join(H, 'raw_5090/formats')
for dim, sizes in (('2d', [512, 2048, 4096, 8192, 16384]), ('3d', [64, 128, 256, 512])):
    for s in sizes:
        C = rd('%s/csr_%s_%d.txt' % (F, dim, s)); HU = rd('%s/handtuned_untuned_%s_%d.txt' % (F, dim, s)); MT = rd('%s/mdh_tuned_%s_%d.txt' % (F, dim, s))
        csr = tm(C, 'csr-scalar'); cus = tm(C, 'cusparse'); ht = ceil(HU); mt = tm(MT, r'MDH \(generated\)')
        N = s * s if dim == '2d' else s ** 3
        bmf = 8 * N; bcsr = (52 if dim == '2d' else 68) * N
        pc = lambda b, t: '%7.1f%%' % (100 * b / (t * 1e-3) / 1792e9)
        print('%-9s %8s %8s %8s %8s | %.2f  %.2f' % ('%s %d' % (dim, s), pc(bcsr, csr), pc(bcsr, cus), pc(bmf, ht), pc(bmf, mt), csr / ht, bcsr / bmf))
print('\n== 2. cross-GPU transfer of the best MDH configuration (stage-1 timings, ratios >= 1)')
def load(p):
    d = {}
    for l in open(p):
        m = re.match(r'\s+stage1 cfg=\[(.*?)\] ([0-9.]+) ms', l)
        if m: d[m.group(1)] = float(m.group(2))
    return d
print('%-9s %-14s %-14s %10s %10s | default: 3050 5090' % ('size', 'best 3050', 'best 5090', '3050cfg@5090', '5090cfg@3050'))
for dim, s in [('2d', 512), ('2d', 2048), ('2d', 4096), ('3d', 64), ('3d', 128), ('3d', 256)]:
    A = load('%s/raw/mdh_tune_%s_%d.txt' % (H, dim, s)); B = load('%s/raw_5090/mdh_tune_%s_%d.txt' % (H, dim, s))
    amin = min(A, key=A.get); bmin = min(B, key=B.get); ud = '16 16 1 0' if dim == '2d' else '8 8 8 1 0'
    print('%-9s %-14s %-14s %10.2f %10.2f |          %.2f %.2f' % ('%s %d' % (dim, s), amin, bmin, B[amin] / B[bmin], A[bmin] / A[amin], A[ud] / A[amin], B[ud] / B[bmin]))
