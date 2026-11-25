#!/usr/bin/env python3
"""Print a compact per-clip timeline from analysis.json: every 0.5 s -> motion, sharpness, face.
Usage: moments.py <folder> [clip ...]"""
import sys, json, os, numpy as np
a = json.load(open(os.path.join(sys.argv[1], "analysis.json")))
names = sys.argv[2:] or sorted(a)
for n in names:
    d = a[n]; fps = d["fps"]; m = np.array(d["motion"]); s = np.array(d["sharp"]); f = np.array(d["face_h"])
    sm = np.median(s)
    row = []
    for k in range(0, len(m), fps // 2):
        seg = slice(k, k + fps // 2)
        mm = m[seg].mean(); ss = s[seg].mean() / sm; ff = f[seg].max()
        c = "F" if ff > 0.04 else "-"
        row.append(f"{k/fps:4.1f}:{mm:3.1f}{c}{'s' if ss < 0.75 else ''}")
    print(f"{n} ({d['dur']:.1f}s, sharp med {sm:.0f})\n   " + " ".join(row))
