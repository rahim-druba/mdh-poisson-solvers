#!/usr/bin/env python3
"""Builds markdown tables from raw/paper_sizes (RTX 3050, paper sizes) and raw_5090/formats (RTX 5090, large sizes)."""
import re, os, sys
H = os.path.dirname(os.path.abspath(__file__))
def rd(p):
    return open(p).read() if os.path.exists(p) else ""
def row(txt, name):
    m = re.search(r'^%s\s+([0-9.]+)\s+\S+\s+(yes|no|NO)' % name, txt, re.M)
    return (float(m.group(1)), m.group(2).lower() == 'yes') if m else (None, False)
def ceil(txt):
    m = re.search(r'HAND-TUNED CEILING:\s+(.+?)\s+([0-9.]+) ms', txt)
    return (m.group(1).strip(), float(m.group(2))) if m else (None, None)
def tcfg(path):
    m = re.search(r'BEST_MDH_TUNED.*cfg=\[(.*?)\]', rd(path)); return m.group(1) if m else '?'
def ppt(path):   # tuned PPCG from the one-session run: (ms, cfg)
    m = re.search(r'^ppcg\s+([0-9.]+) ms', rd(path), re.M); c = re.search(r'cfg=\[(.*?)\]', rd(path))
    return (float(m.group(1)), c.group(1).replace(' | ', ' ; ')) if m else (None, '?')
def ppsweep(path):   # 5090: (tuned ms, cfg, default ms)
    t = rd(path); b = re.search(r'BEST_PPCG_TUNED\s+([0-9.]+) ms cfg=\[(.*?)\]', t); d = re.search(r'PPCG_DEFAULT ppcg\s+([0-9.]+) ms', t)
    return (float(b.group(1)) if b else None, b.group(2).replace(' | ', ' ; ') if b else '?', float(d.group(1)) if d else None)
def f(x): return '-' if x is None else '%.5f' % x
def r(a, b): return '-' if a is None or b is None else '%.2f' % (a / b)
out = []
P = os.path.join(H, 'raw/paper_sizes')
for dim, sizes, lab in (('2d', [('512', '16x32'), ('1024', '32x32'), ('2048', '32x64'), ('4096', '64x64')], 'N (grid)'),
                        ('3d', [('8', '8'), ('16', '16'), ('24', '24'), ('32', '32')], 'M')):
    out.append('\n### RTX 3050, paper sizes, %s matvec (ms; one session, 200 launches x 9)\n' % dim.upper())
    out.append('| %s | CSR | cuSPARSE | PPCG (default) | PPCG tuned | PPCG tuned cfg | MDH untuned | MDH tuned | tuned cfg | hand-tuned | best HT cfg | untuned/HT | tuned/HT | CSR/HT | PPCG default/HT | PPCG tuned/HT | PPCG tuned/MDH tuned | all correct |' % lab)
    out.append('|' + '---|' * 18)
    for key, g in sizes:
        U = rd('%s/matvec_%s_%s_untuned.txt' % (P, dim, key)); T = rd('%s/matvec_%s_%s_tuned.txt' % (P, dim, key))
        HT = rd('%s/handtuned_%s_%s.txt' % (P, dim, key))
        csr, a = row(U, 'sparse'); cus, b = row(U, 'cusparse'); pp, c = row(U, 'ppcg'); mu, d = row(U, 'mdh')
        mt, e = row(T, 'mdh'); name, ht = ceil(HT)
        cfgf = '%s/mdh_tune_%s_%s.txt' % (P, dim, g if dim == '2d' else key)
        ok = all([a, b, c, d, e])
        pt, pcfg = ppt('%s/ppcg_tuned_%s_%s.txt' % (P, dim, key))
        out.append('|' + ' %s |' * 18 % (
            key if dim == '3d' else '%s (%s)' % (key, g), f(csr), f(cus), f(pp), f(pt), pcfg, f(mu), f(mt), tcfg(cfgf), f(ht), name,
            r(mu, ht), r(mt, ht), r(csr, ht), r(pp, ht), r(pt, ht), r(pt, mt), 'yes' if ok else 'CHECK'))
F = os.path.join(H, 'raw_5090/formats')
for dim, sizes, lab in (('2d', ['512', '2048', '4096', '8192', '16384'], 'side'), ('3d', ['64', '128', '256', '512'], 'side')):
    out.append('\n### RTX 5090, large sizes, %s matvec (ms; one session)\n' % dim.upper())
    out.append('| %s | CSR | cuSPARSE | MDH untuned | MDH tuned | hand-tuned | best HT cfg | PPCG default | PPCG tuned | PPCG tuned cfg | untuned/HT | tuned/HT | CSR/HT | cuSPARSE/HT | PPCG default/HT | PPCG tuned/HT | PPCG tuned/MDH tuned |' % lab)
    out.append('|' + '---|' * 17)
    for s in sizes:
        C = rd('%s/csr_%s_%s.txt' % (F, dim, s)); HU = rd('%s/handtuned_untuned_%s_%s.txt' % (F, dim, s)); MT = rd('%s/mdh_tuned_%s_%s.txt' % (F, dim, s))
        csr = re.search(r'csr-scalar\s+([0-9.]+) ms', C); cus = re.search(r'cusparse\s+([0-9.]+) ms', C)
        csr = float(csr.group(1)) if csr else None; cus = float(cus.group(1)) if cus else None
        mu = re.search(r'MDH \(generated\)\s+([0-9.]+) ms', HU); mt = re.search(r'MDH \(generated\)\s+([0-9.]+) ms', MT)
        mu = float(mu.group(1)) if mu else None; mt = float(mt.group(1)) if mt else None
        name, ht = ceil(HU)
        pt, pcfg, pd = ppsweep(os.path.join(H, 'raw_5090/ppcg/ppcg_%s_%s.txt' % (dim, s)))
        out.append('|' + ' %s |' * 17 % (s, f(csr), f(cus), f(mu), f(mt), f(ht), name, f(pd), f(pt), pcfg, r(mu, ht), r(mt, ht), r(csr, ht), r(cus, ht), r(pd, ht), r(pt, ht), r(pt, mt)))
open(os.path.join(H, 'results_paper_sizes.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
