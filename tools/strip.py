#!/usr/bin/env python3
"""Dense strip of one clip window for picking exact in-points.
Usage: strip.py clip.MOV start end fps out.jpg"""
import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from reelkit import read_frames
c, a, b, fps, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
fr = list(read_frames(c, a, b - a, w=180, h=320, fps=fps))
tiles = []
for i, f in enumerate(fr):
    f = f[:, :, ::-1].copy(); cv2.putText(f, f"{a + i / fps:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1); tiles.append(f)
per = 12; rows = []
for r in range(0, len(tiles), per):
    row = tiles[r:r + per]; row += [np.zeros_like(tiles[0])] * (per - len(row)); rows.append(np.hstack(row))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
