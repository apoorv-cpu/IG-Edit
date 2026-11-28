#!/usr/bin/env python3
"""Per-frame global camera translation for every clip in a folder -> <dir>/shake.json.
shake.py clips_dir   ;  window score: jitter(t0, dur) = mean |d - smooth(d)| in px @270w."""
import sys,os,json,cv2,numpy as np
def track(path):
    cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS); prev=None; out=[]
    win=None
    while True:
        ok,f=cap.read()
        if not ok: break
        if f.shape[1]>f.shape[0]: f=cv2.rotate(f,cv2.ROTATE_90_CLOCKWISE)
        g=cv2.cvtColor(cv2.resize(f,(270,480)),cv2.COLOR_BGR2GRAY).astype(np.float32)
        if win is None: win=cv2.createHanningWindow(g.shape[::-1],cv2.CV_32F)
        if prev is not None:
            (dx,dy),r=cv2.phaseCorrelate(prev,g,win); out.append([round(dx,3),round(dy,3)])
        prev=g
    return fps,out
def jitter(d,fps,t0,dur):
    a=np.array(d[int(t0*fps):int((t0+dur)*fps)])
    if len(a)<10: return 99
    k=max(5,int(fps/6))|1
    sm=np.array([np.convolve(np.pad(a[:,i],k//2,mode='edge'),np.ones(k)/k,'valid') for i in (0,1)]).T
    return float(np.mean(np.linalg.norm(a-sm,axis=1)))
if __name__=="__main__":
    d=sys.argv[1]; res={}
    for c in sorted(x for x in os.listdir(d) if x.upper().endswith('.MOV')):
        fps,o=track(os.path.join(d,c)); res[c]={'fps':fps,'d':o}; print(c,len(o),flush=True)
    json.dump(res,open(os.path.join(d,'shake.json'),'w'))

def wobble(d,fps,t0,dur,win=0.5):
    """Low-frequency irregularity: cumulative camera path vs its 0.5 s moving average (px @270w)."""
    a=np.array(d[int(t0*fps):int((t0+dur)*fps)])
    if len(a)<10: return 99
    p=np.cumsum(a,axis=0); k=max(5,int(fps*win))|1
    sm=np.array([np.convolve(np.pad(p[:,i],k//2,mode='edge'),np.ones(k)/k,'valid') for i in (0,1)]).T
    return float(np.mean(np.linalg.norm(p-sm,axis=1)))
