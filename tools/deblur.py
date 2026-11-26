#!/usr/bin/env python3
"""Find sharper in-points for blurry shots: deblur.py plan.json [--write]
Uses <clips_dir>/analysis.json (15 fps motion/sharp/face curves). A shot is blurry when its 15th-percentile
sharpness is below 0.6 x the clip's 90th percentile. Candidates: same clip first, then other clips, same speed,
moving at the first frame, no overlap with other shots; the opener must show a face."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from render import snap_cuts
plan = json.load(open(sys.argv[1])); song = json.load(open(plan["song_json"]))
A = json.load(open(os.path.join(plan["clips_dir"], "analysis.json")))
cuts = snap_cuts(plan, song)
spans = [(s["clip"], s["in"], s["in"] + (cuts[i + 1] - cuts[i]) * s["speed"]) for i, s in enumerate(plan["shots"])]

def score(clip, a, b):
    c = A[clip]; f = c["fps"]; sh = np.array(c["sharp"]); i, j = int(a * f), int(np.ceil(b * f))
    if j > len(sh) - 2 or i < 1: return None
    return float(np.percentile(sh[i:j], 15) / np.percentile(sh, 90))

def free(clip, a, b, skip):
    return all(not (c == clip and a < d - 0.05 and x < b - 0.05) for k, (c, x, d) in enumerate(spans) if k != skip)

changed = False
for i, s in enumerate(plan["shots"]):
    clip, a, b = spans[i]; L = b - a; cur = score(clip, a, b) or 0
    if cur >= 0.6: continue
    best = None
    for c in [clip] + [k for k in A if k != clip]:
        mot = np.array(A[c]["motion"]); f = A[c]["fps"]; face = np.array(A[c]["face"])
        for t in np.arange(0.2, A[c]["dur"] - L - 0.2, 0.1):
            sc = score(c, t, t + L)
            if sc is None or sc < 0.75 or not free(c, t, t + L, i): continue
            k = int(t * f)
            if mot[k] < 0.5 * np.median(mot): continue
            if i == 0 and face[k] <= 0: continue
            pen = 0 if c == clip else 0.15          # prefer same clip (same action)
            val = sc - pen - (0.02 * abs(t - a) if c == clip else 0)
            if best is None or val > best[0]: best = (val, c, round(float(t), 1), sc)
        if best and best[1] == clip: break
    print(f"shot {i:2d} {clip} in {a:.2f} sharp {cur:.2f}  ->  {best[1:] if best else 'no sharper window'}  ({s.get('move','')})")
    if best and "--write" in sys.argv:
        s["clip"], s["in"] = best[1], best[2]; spans[i] = (best[1], best[2], best[2] + L); changed = True
if changed: json.dump(plan, open(sys.argv[1], "w"), indent=1)
