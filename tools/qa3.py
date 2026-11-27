import sys,subprocess,re,numpy as np,cv2
f,rep=sys.argv[1],sys.argv[2]
cuts=eval(re.search(r"cuts \(song time\): (\[.*\])",open(rep).read()).group(1)); cuts=[c-cuts[0] for c in cuts]
p=subprocess.Popen(['ffmpeg','-v','error','-i',f,'-vf','scale=540:960','-f','rawvideo','-pix_fmt','gray','-'],stdout=subprocess.PIPE)
fr=[]
while True:
    b=p.stdout.read(540*960)
    if len(b)<540*960: break
    fr.append(np.frombuffer(b,np.uint8).reshape(960,540))
g=[x.astype(np.float32) for x in fr]
d=np.array([0]+[np.mean(np.abs(cv2.resize(g[i],(270,480))-cv2.resize(g[i-1],(270,480)))) for i in range(1,len(g))])/2.55
sh=np.array([cv2.Laplacian(x,cv2.CV_32F).var() for x in g])
clip=np.array([np.mean(x>=250)*100 for x in fr])
ok=True
for k in range(len(cuts)-1):
    a,b=int(round(cuts[k]*30))+1,int(round(cuts[k+1]*30))-1
    m=d[a:b]; s=sh[a+1:b]; from scipy.ndimage import maximum_filter1d; ref=maximum_filter1d(s,17)
    win=np.convolve(m,np.ones(15)/15,'valid'); mn=win.min() if len(win) else m.mean()
    blur=np.mean(s<0.8*ref)*100; worst=(s/ref).min()
    flag=[]
    if mn<0.6: flag.append('STILL')
    if worst<0.8: flag.append('BLUR')
    if clip[a-1:b].mean()>1.5: flag.append('CLIP')
    if flag: ok=False
    print(f"shot {k}: {cuts[k]:5.2f}-{cuts[k+1]:5.2f}s motion min0.5s {mn:4.2f} mean {m.mean():4.2f}  frames<0.8sharp {blur:4.1f}% worst {worst:.2f}  clipped {clip[a-1:b].mean():.2f}%  {' '.join(flag)}")
print('identical consecutive frames at:',[round(i/30,2) for i in np.where(d<0.05)[0] if i>0],' ALL OK' if ok else ' CHECK')
