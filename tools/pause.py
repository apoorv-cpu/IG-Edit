import sys,subprocess,re,numpy as np,cv2
f,rep=sys.argv[1],sys.argv[2]
txt=open(rep).read()
cuts=eval(re.search(r"cuts \(song time\): (\[.*\])",txt).group(1)); cuts=[c-cuts[0] for c in cuts]
names=re.findall(r"shot +\d+ (DSC_\d+)\.MOV in ([\d.]+).*?  (\S.*)$",txt,re.M)
p=subprocess.Popen(['ffmpeg','-v','error','-i',f,'-vf','scale=135:240','-f','rawvideo','-pix_fmt','gray','-'],stdout=subprocess.PIPE)
fr=[]
while True:
    b=p.stdout.read(135*240)
    if len(b)<135*240: break
    fr.append(np.frombuffer(b,np.uint8).reshape(240,135).astype(np.float32))
d=np.array([0]+[np.mean(np.abs(fr[i]-fr[i-1]))/2.55 for i in range(1,len(fr))])
for k in range(len(cuts)-1):
    a,b=int(round(cuts[k]*30))+1,int(round(cuts[k+1]*30))-1
    m=np.convolve(d[a:b],np.ones(9)/9,'same')  # 0.3s smoothing
    prof=' '.join(f"{x:.1f}" for x in m[::6])  # every 0.2s
    med=np.median(m); lo=m.min()
    print(f"shot {k} {names[k][0]} in {names[k][1]} {cuts[k]:.2f}-{cuts[k+1]:.2f} med {med:.1f} min {lo:.1f} at {(np.argmin(m)+a)/30:.2f}s | {names[k][2][:40]}\n   {prof}")
