import json,subprocess,sys,urllib.parse,statistics
U='17841425085791431'
def get(b,after=None):
    f=f"business_discovery.username({b}){{followers_count,media.limit(50)"+(f".after({after})" if after else "")+"{media_type,media_product_type,like_count,comments_count,timestamp,permalink,media_url,caption}}"
    u=f"https://graph.facebook.com/v21.0/{U}?fields="+urllib.parse.quote(f,safe='(){},.')
    return json.loads(subprocess.run(['curl','-sg',u],capture_output=True,text=True).stdout)
out={}
for b in sys.argv[1:]:
    items=[];after=None
    for _ in range(4):
        d=get(b,after)
        if 'business_discovery' not in d: print(b,d.get('error',{}).get('message')); break
        m=d['business_discovery']['media']; items+=m['data']
        after=m.get('paging',{}).get('cursors',{}).get('after')
        if not after: break
    reels=[x for x in items if x.get('media_product_type')=='REELS' and x['timestamp']>='2026-08-01']
    out[b]=reels
    if reels:
        lk=[x.get('like_count',0) for x in reels]; md=statistics.median(lk)
        print(b,len(items),'reels since Aug:',len(reels),'median likes',md,'top',sorted(lk)[-5:])
json.dump(out,open('media.json','w'))
