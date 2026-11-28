import cv2,numpy as np,sys
p=sys.argv[1]; out=sys.argv[2]; ts=[float(x) for x in sys.argv[3].split(',')]
cap=cv2.VideoCapture(p); ims=[]
for t in ts:
    cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,f=cap.read()
    if not ok: continue
    f=cv2.resize(f,(216,384)); cv2.putText(f,f"{t:.1f}",(5,20),0,0.6,(0,255,255),2); ims.append(f)
rows=[np.hstack(ims[i:i+8]+[np.zeros_like(ims[0])]*(8-len(ims[i:i+8]))) for i in range(0,len(ims),8)]
cv2.imwrite(out,np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,80])
