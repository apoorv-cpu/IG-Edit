#!/usr/bin/env python3
"""Contact sheet per folder: one row per clip, one thumb every STEP seconds, labelled.
Usage: overview.py <folder_with_MOVs> <out.jpg> [step=1.0]"""
import sys, os, glob, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from reelkit import read_frames, probe

src, out = sys.argv[1], sys.argv[2]
step = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
TW, TH = 135, 240
rows = []
for f in sorted(glob.glob(os.path.join(src, "*.MOV"))):
    d = probe(f)["dur"]
    thumbs = list(read_frames(f, 0, d, w=TW, h=TH, fps=1 / step))
    row = np.full((TH + 18, 18 * TW, 3), 20, np.uint8)
    for i, t in enumerate(thumbs[:18]):
        row[18:, i * TW:(i + 1) * TW] = t[:, :, ::-1]
        cv2.putText(row, f"{i*step:.0f}", (i * TW + 3, 32), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
    cv2.putText(row, f"{os.path.basename(f)}  {d:.1f}s", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    rows.append(row)
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
