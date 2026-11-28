#!/usr/bin/env python3
"""Review strip for a plan: one row per shot, 6 frames across the shot (0.5x source window). planstrip.py plan.json out.jpg"""
import sys, os, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from reelkit import read_frames
p = json.load(open(sys.argv[1])); rows = []
for i, s in enumerate(p["shots"]):
    src = s["len"] * s["speed"]; path = os.path.join(p["clips_dir"], s["clip"])
    fr = list(read_frames(path, s["in"], src, w=162, h=288, fps=6 / src))[:6]
    row = np.full((300, 6 * 162 + 150, 3), 25, np.uint8)
    for k, f in enumerate(fr): row[12:, 150 + k * 162:150 + (k + 1) * 162] = f[:, :, ::-1]
    for j, txt in enumerate([f"#{i}", s["clip"][4:8], f"in {s['in']:.2f}", f"{s['len']:.1f}s"]):
        cv2.putText(row, txt, (6, 40 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    rows.append(row)
cv2.imwrite(sys.argv[2], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
