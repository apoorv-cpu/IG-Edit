# For each shot (0.5x), read native frames in [in-0.6, in+len/2+0.6], local sharpness ratio vs +-8 frame max,
# and pick the longest window (>= 70% planned) with all frames >= TH that overlaps the planned window most.
import sys,json,os,numpy as np,cv2
sys.path.insert(0,'/home/user/IG-Edit/tools')
from reelkit import read_frames,probe
from scipy.ndimage import maximum_filter1d
TH=float(sys.argv[2]) if len(sys.argv)>2 else 0.82
plan=json.load(open(sys.argv[1])); FPS=59.94
for i,s in enumerate(plan['shots']):
    path=os.path.join(plan['clips_dir'],s['clip']); dur=probe(path)['dur']
    a0=max(0,s['in']-0.6); b0=min(dur-0.25,s['in']+s['len']/2+0.6)
    fr=list(read_frames(path,a0,b0-a0,w=540,h=960,fps=None))
    sh=np.array([cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY).astype(np.float32),cv2.CV_32F).var() for f in fr])
    g=[cv2.cvtColor(cv2.resize(f,(270,480)),cv2.COLOR_RGB2GRAY).astype(np.float32) for f in fr]
    mo=np.array([0]+[np.mean(np.abs(g[k]-g[k-1]))/2.55 for k in range(1,len(g))])
    r=sh/maximum_filter1d(sh,17)
    good=r>=TH
    # runs
    runs=[];k=0
    while k<len(good):
        if good[k]:
            j=k
            while j<len(good) and good[j]: j+=1
            runs.append((k,j)); k=j
        else: k+=1
    pa,pb=(s['in']-a0)*FPS,(s['in']-a0+s['len']/2)*FPS
    best=max(runs,key=lambda q:min(q[1],pb)-max(q[0],pa)) if runs else None
    bad=[(round(a0+k/FPS,2),round(float(r[k]),2)) for k in range(int(pa),min(int(pb),len(r))) if r[k]<TH]
    if best:
        ba,bb=a0+best[0]/FPS,a0+best[1]/FPS
        print(f"shot {i} {s['clip']} plan {s['in']:.2f}-{s['in']+s['len']/2:.2f}  bad frames in plan: {len(bad)} {bad[:6]}  best sharp run {ba:.2f}-{bb:.2f} ({bb-ba:.2f}s src)  motion med {np.median(mo[best[0]:best[1]]):.1f}  {s['move']}")
