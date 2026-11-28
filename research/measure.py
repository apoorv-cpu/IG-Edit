import cv2,numpy as np,sys,json
def measure(path):
    cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS) or 30
    prev=None;ph=None;cuts=[0];tx=[];diffs=[];i=0
    while True:
        ok,f=cap.read()
        if not ok: break
        g=cv2.cvtColor(cv2.resize(f,(180,320)),cv2.COLOR_BGR2GRAY).astype(np.float32)
        h=cv2.calcHist([cv2.resize(f,(90,160))],[0,1,2],None,[8,8,8],[0,256]*3); h=cv2.normalize(h,h).flatten()
        if prev is not None:
            hd=cv2.compareHist(ph,h,cv2.HISTCMP_BHATTACHARYYA); d=float(np.mean(np.abs(g-prev)))
            loc=np.median(diffs[-10:]) if diffs else 1
            if (hd>0.35 or (d>10 and d>4*loc)) and i-cuts[-1]>4: cuts.append(i); tx.append(None)
            else:
                (dx,dy),_=cv2.phaseCorrelate(prev,g); tx.append((dx,dy))
            diffs.append(d)
        prev=g;ph=h;i+=1
    n=i; cuts.append(n)
    shots=np.diff(cuts)/fps
    # shake: per shot, high-freq jitter of global translation
    jit=[];seg=[]
    for t in tx+[None]:
        if t is None:
            if len(seg)>8:
                a=np.array(seg); sm=np.array([np.convolve(a[:,k],np.ones(7)/7,'same') for k in (0,1)]).T
                jit.append(float(np.mean(np.linalg.norm((a-sm)[3:-3],axis=1))))
            seg=[]
        else: seg.append(t)
    return dict(file=path.split('/')[-1],dur=round(n/fps,1),fps=round(fps,2),shots=len(shots),avg_shot=round(float(np.mean(shots)),2),
                med_shot=round(float(np.median(shots)),2),first_shot=round(float(shots[0]),2),motion=round(float(np.median(diffs)),2),
                jitter=round(float(np.median(jit)),3) if jit else None, jitter_max=round(float(max(jit)),3) if jit else None)
for p in sys.argv[1:]:
    print(json.dumps(measure(p)),flush=True)
