#!/usr/bin/env python3
"""Per-shot sharpness report for a plan: blurcheck.py plan.json
Sharpness = variance of Laplacian on a 540x960 grey frame. Flags shots whose frames drop well below the reel median."""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from render import snap_cuts, shot_frames
plan = json.load(open(sys.argv[1])); song = json.load(open(plan["song_json"]))
cuts = snap_cuts(plan, song); rows = []
for i, s in enumerate(plan["shots"]):
    fr = shot_frames(os.path.join(plan["clips_dir"], s["clip"]), s["in"], cuts[i + 1] - cuts[i], s["speed"])
    sh = [cv2.Laplacian(cv2.cvtColor(cv2.resize(f, (540, 960), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY), cv2.CV_32F).var() for f in fr]
    rows.append((i, s, np.array(sh)))
med = np.median(np.concatenate([r[2] for r in rows]))
for i, s, sh in rows:
    bad = np.mean(sh < 0.6 * med)
    print(f"shot {i:2d} {s['clip']} in {s['in']:.2f} x{s['speed']}  median {np.median(sh)/med:.2f}  min {sh.min()/med:.2f}  blurry {bad*100:3.0f}%  {s.get('move','')}{'  <-- BLUR' if bad > 0.2 else ''}")
