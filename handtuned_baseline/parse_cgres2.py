import re,sys
names={0:'hand-tuned',3:'MDH tuned',4:'PPCG tuned',1:'CSR',2:'cuSPARSE'}
def parse(path):
    d={}; cfg=''
    for l in open(path):
        if l.startswith('#'): cfg=l.strip()
        m=re.match(r'cg-resident MV=(\d)\s+([0-9.]+) ms/iter \(\+-([0-9.]+)%\)\s+residual norm after \d+ iterations: ([0-9.e+-]+)',l)
        if m: d[int(m.group(1))]=(float(m.group(2)),float(m.group(3)),float(m.group(4)))
    return d,cfg
def show(dirn,dim_sizes):
    print('%-9s | %9s %9s %9s %9s %9s | CSR/MDH cuS/MDH MDH/HT PPCG/MDH | resid spread (max-min)/min | max noise'%('size','HT','MDHt','PPCGt','CSR','cuSPARSE'))
    for dim,sizes in dim_sizes:
        for s in sizes:
            try: d,_=parse('%s/cgres_%s_%d.txt'%(dirn,dim,s))
            except Exception: continue
            if len(d)<5: continue
            r=[d[k][2] for k in d]; sp=(max(r)-min(r))/min(r)*100
            print('%-9s | %9.4f %9.4f %9.4f %9.4f %9.4f | %.2f %.2f %.2f %.2f | %.2f%% | %.1f%%'%('%s %d'%(dim,s),d[0][0],d[3][0],d[4][0],d[1][0],d[2][0],d[1][0]/d[3][0],d[2][0]/d[3][0],d[3][0]/d[0][0],d[4][0]/d[3][0],sp,max(d[k][1] for k in d)))
if __name__=='__main__':
    show(sys.argv[1],[('2d',[512,2048,4096,8192,16384]),('3d',[64,128,256,512])] if '5090' in sys.argv[1] else [('2d',[512,2048,4096]),('3d',[64,128,256])])
