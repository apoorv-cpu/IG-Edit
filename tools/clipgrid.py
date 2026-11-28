#!/usr/bin/env python3
"""clipgrid.py clips_dir out.jpg [step]  one row per clip, frame every `step` s, labelled."""
import sys,os,cv2,numpy as np
sys.path.insert(0,os.path.dirname(__file__))
from reelkit import read_frames
d,out=sys.argv[1],sys.argv[2]; step=float(sys.argv[3]) if len(sys.argv)>3 else 1.0
rows=[]
for c in sorted(x for x in os.listdir(d) if x.upper().endswith('.MOV')):
    ims=[];k=0
    for f in read_frames(os.path.join(d,c),0,60,fps=1/step):
        f=cv2.resize(f,(96,170))[:,:,::-1].copy(); cv2.putText(f,f"{k*step:.0f}",(3,14),0,0.45,(0,255,255),1); ims.append(f); k+=1
        if k>=16: break
    lab=np.zeros((170,60,3),np.uint8); cv2.putText(lab,c[4:8],(4,90),0,0.5,(255,255,255),1)
    rows.append(np.hstack([lab]+ims+[np.zeros((170,96,3),np.uint8)]*(16-len(ims))))
cv2.imwrite(out,np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,82])
