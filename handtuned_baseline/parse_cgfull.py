import re,sys
names={'D':'fused HW','E':'fused HW + MDH stencil','A':'unfused HW','B':'unfused HW + MDH stencil','C':'fully MDH'}
def parse(path):
    d={}
    for l in open(path):
        m=re.match(r'([DEABCFG]) cg-resident MV=(-?\d) VEC=(-?\d)\s+([0-9.]+) ms/iter \(\+-([0-9.]+)%\)\s+residual norm after \d+ iterations: ([0-9.e+-]+)',l)
        if m: d[m.group(1)]=(float(m.group(4)),float(m.group(5)),float(m.group(6)))
    return d
def show(dirn,ds):
    print('%-9s | %9s %9s %9s %9s %9s | C/A (gen vs HW unfused) C/B C/D (vs fused) E/D | resid spread | noise'%('size','D fusedHW','E fus+MDHst','A unfHW','B unf+MDHst','C fullMDH'))
    for dim,sizes in ds:
        for s in sizes:
            try: d=parse('%s/cgfull_%s_%d.txt'%(dirn,dim,s))
            except Exception: continue
            if len(d)<7: print(dim,s,'INCOMPLETE',sorted(d)); continue
            r=[d[k][2] for k in d]; sp=(max(r)-min(r))/min(r)*100
            print('%-9s | %9.4f %9.4f %9.4f %9.4f %9.4f | %.2f %.2f %.2f %.2f | %.2f %.2f %.2f %.2f | %.2f%% | %.1f%%'%('%s %d'%(dim,s),d['D'][0],d['E'][0],d['A'][0],d['B'][0],d['C'][0],d['C'][0]/d['A'][0],d['C'][0]/d['B'][0],d['C'][0]/d['D'][0],d['E'][0]/d['D'][0],d['F'][0]/d['C'][0],d['G'][0]/d['C'][0],d['F'][0]/d['D'][0],d['G'][0]/d['D'][0],sp,max(d[k][1] for k in d)))
if __name__=='__main__':
    show(sys.argv[1],[('2d',[512,2048,4096,8192,16384]),('3d',[64,128,256,512])] if '5090' in sys.argv[1] else [('2d',[512,2048,4096]),('3d',[64,128,256])])
