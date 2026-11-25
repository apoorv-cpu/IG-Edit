#!/usr/bin/env python3
"""Per-clip motion / sharpness / face curves at 15 fps (on a 270x480 upright proxy).
Usage: analyze.py <folder> -> writes <folder>/analysis.json
motion: mean abs luma change between consecutive proxy frames (x100, 0..~10)
sharp:  Laplacian variance of the centre region
face:   largest face score + box height (fraction of frame) via MediaPipe BlazeFace"""
import sys, os, glob, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from reelkit import read_frames, probe, MODELS
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "blaze_face_short_range.tflite")),
    min_detection_confidence=0.5))
FPS = 15
out = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.MOV"))):
    d = probe(f)["dur"]; prev = None
    mo, sh, fs, fh = [], [], [], []
    for fr in read_frames(f, 0, d, w=270, h=480, fps=FPS):
        g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mo.append(0.0 if prev is None else float(np.mean(np.abs(g - prev)) / 2.55)); prev = g
        sh.append(float(cv2.Laplacian(g[60:420, 40:230], cv2.CV_32F).var()))
        r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr)))
        if r.detections:
            b = max(r.detections, key=lambda x: x.bounding_box.height)
            fs.append(round(b.categories[0].score, 2)); fh.append(round(b.bounding_box.height / 480, 3))
        else:
            fs.append(0.0); fh.append(0.0)
    out[os.path.basename(f)] = {"dur": d, "fps": FPS, "motion": [round(x, 2) for x in mo],
                                "sharp": [round(x, 1) for x in sh], "face": fs, "face_h": fh}
    print(os.path.basename(f), f"{d:.1f}s motion med {np.median(mo):.2f} faces {np.mean(np.array(fs)>0):.0%}", flush=True)
json.dump(out, open(os.path.join(sys.argv[1], "analysis.json"), "w"))
