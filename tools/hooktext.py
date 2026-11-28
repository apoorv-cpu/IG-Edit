"""Render an on-screen hook (white bold text with soft shadow) as a transparent PNG."""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap
FONT="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
def make(text, out, W=2160, H=3840, y=0.13, size=140, width=20):
    img=Image.new("RGBA",(W,H),(0,0,0,0)); f=ImageFont.truetype(FONT,size)
    lines=[]
    for para in text.split("\n"): lines+=textwrap.wrap(para,width) or [""]
    lh=int(size*1.22); y0=int(H*y)
    sh=Image.new("RGBA",(W,H),(0,0,0,0)); ds=ImageDraw.Draw(sh); d=ImageDraw.Draw(img)
    for i,l in enumerate(lines):
        w=d.textlength(l,font=f); x=(W-w)/2; yy=y0+i*lh
        ds.text((x+4,yy+6),l,font=f,fill=(0,0,0,170))
        d.text((x,yy),l,font=f,fill=(255,255,255,255))
    sh=sh.filter(ImageFilter.GaussianBlur(10))
    Image.alpha_composite(sh,img).save(out)
if __name__=="__main__":
    import sys; make(sys.argv[1],sys.argv[2])
