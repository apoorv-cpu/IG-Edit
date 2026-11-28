import sys,ast,subprocess
out=subprocess.run(['python3','/home/user/IG-Edit/tools/autoplan.py']+sys.argv[1:],capture_output=True,text=True).stdout
best={}
for l in out.splitlines():
    c=ast.literal_eval(l); k=(c['clip'],round(c['a']*2)/2)
    if k not in best or c['score']>best[k]['score']: best[k]=c
for k in sorted(best): c=best[k]; print(c['clip'][4:8],c['a'],c['L'],c['score'],'med',c['med'],'face',c['face'],'sr',c['sr'])
