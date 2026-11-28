#!/usr/bin/env python3
"""Draft a reel plan from analysis.json: autoplan.py cfg.json -> plan json on stdout.
cfg: name, clips_dir, song, song_start, kind ('model'|'mannequin'), total (s), ban {clip: [[a,b],...]},
     force_open {clip, in}, exclude [clips], grade {...}
All shots 0.5x. A window qualifies when motion never dips (no hesitation), it starts moving,
it is sharp (15th pct >= 0.75 of the clip's p90) and, for model reels, the face is visible."""
import sys, json, os, numpy as np
cfg = json.load(open(sys.argv[1]))
A = json.load(open(os.path.join(cfg["clips_dir"], "analysis.json")))
song = json.load(open(cfg["song"].replace(".wav", ".json")))
beats = np.array(song["beats"]); t0 = float(beats[np.argmin(abs(beats - cfg["song_start"]))])
model = cfg.get("kind", "model") == "model"
ban = cfg.get("ban", {}); excl = set(cfg.get("exclude", []))
cands = []
for clip, d in A.items():
    if clip in excl: continue
    f = d["fps"]; m = np.array(d["motion"]); s = np.array(d["sharp"]); fh = np.array(d["face_h"])
    ms = np.convolve(m, np.ones(5) / 5, "same"); p90 = np.percentile(s, 90)
    n = len(m)
    for L in (1.8, 1.6, 1.4, 1.2):
        w = int(L * f)
        for i in range(2, n - w - 3, 3):
            a, b = i / f, (i + w) / f
            if any(a < y and x < b for x, y in ban.get(clip, [])): continue
            seg = ms[i:i + w]; med = np.median(seg)
            if seg.min() < max(0.45, 0.4 * med) or m[i:i + 4].mean() < 0.7: continue
            sr = np.percentile(s[i:i + w], 15) / p90
            if sr < 0.75: continue
            face = np.mean(fh[i:i + w] > 0.035)
            if model and face < cfg.get("min_face", 0.0): continue
            sc = sr + 0.15 * min(med, 4) + (0.5 * face if model else 0) + 0.15 * L
            cands.append(dict(clip=clip, a=round(a, 2), L=L, score=round(float(sc), 3), med=round(float(med), 2),
                              face=round(float(fh[i:i + w].max()), 3), sr=round(float(sr), 2)))
cands.sort(key=lambda c: -c["score"])
if len(sys.argv) > 2:
    for c in cands:
        if c["clip"] in sys.argv[2:]: print(c)
    raise SystemExit
total = cfg.get("total", 24.0); used = {}; shots = []; t = 0
fo = cfg.get("force_open")
if fo:
    c = next((c for c in cands if c["clip"] == fo["clip"] and abs(c["a"] - fo["in"]) < 0.25), None) or dict(clip=fo["clip"], a=fo["in"], L=fo.get("L", 1.6), score=0)
    shots.append(c); used[c["clip"]] = [(c["a"], c["a"] + c["L"])]; t += c["L"] * 2
per_clip = cfg.get("per_clip", 1)
for c in cands:
    if t >= total - 1.0: break
    u = used.get(c["clip"], [])
    if len(u) >= per_clip: continue
    if any(c["a"] < y + 1.0 and x - 1.0 < c["a"] + c["L"] for x, y in u): continue
    L = min(c["L"], max(1.2, (total - t) / 2))
    c = dict(c, L=round(L, 2)); shots.append(c); used.setdefault(c["clip"], []).append((c["a"], c["a"] + L)); t += L * 2
plan = dict(name=cfg["name"], clips_dir=cfg["clips_dir"], song=cfg["song"], song_json=cfg["song"].replace(".wav", ".json"),
            song_start=round(t0, 3), snap="beats", denoise="hqdn3d=1.0:0.8:0:0", min_start_motion=0.6, only_4k=True,
            grade=cfg.get("grade", {"sat_cap": 1.0, "p50": 150, "gamma_range": [1.0, 1.0], "strength": 1.0}),
            shots=[dict(clip=c["clip"], **{"in": c["a"]}, speed=0.5, len=round(c["L"] * 2, 2),
                        move=f"auto score {c['score']} med {c.get('med')} face {c.get('face')} sr {c.get('sr')}") for c in shots],
            end_fade=1.5)
print(json.dumps(plan, indent=1))
