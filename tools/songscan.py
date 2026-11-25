#!/usr/bin/env python3
"""Scan a song: tempo, beats, loudness curve, and candidate reel windows.
Usage: songscan.py song.wav [window_s=27] -> prints top windows + writes song.json
Windows start on a detected beat that follows an energy rise (section start),
ranked by mean loudness + onset density (chorus/hook proxy)."""
import sys, json, numpy as np, librosa


def track_beats(onset, fps, lo=70, hi=180, tight=100.0):
    """Ellis-style DP beat tracker in plain numpy (librosa's numba tracker segfaults here)."""
    o = onset - onset.mean(); o /= (o.std() + 1e-9)
    ac = np.correlate(o, o, 'full')[len(o) - 1:]
    lags = np.arange(len(ac)); bpm = 60 * fps / np.maximum(lags, 1)
    ok = (bpm >= lo) & (bpm <= hi)
    # log-gaussian weighting around 115 bpm
    w = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1) / 115) / 0.9) ** 2)
    lag = int(lags[ok][np.argmax((ac * w)[ok])])
    period = lag; n = len(o)
    score = o.copy(); back = -np.ones(n, int)
    for i in range(n):
        a, b = i - 2 * period, i - period // 2
        if b <= 0: continue
        a = max(a, 0); prev = np.arange(a, b)
        pen = -tight * (np.log((i - prev) / period)) ** 2
        cand = score[prev] + pen
        j = int(np.argmax(cand)); score[i] = o[i] + cand[j]; back[i] = prev[j]
    # end at best score in final period
    i = int(np.argmax(score[-period:])) + n - period
    beats = []
    while i >= 0:
        beats.append(i); i = back[i]
    return 60 * fps / period, np.array(beats[::-1])

path = sys.argv[1]; win = float(sys.argv[2]) if len(sys.argv) > 2 else 27.0
y, sr = librosa.load(path, sr=22050, mono=True)
hop = 512
onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
tempo, beats = track_beats(onset, sr / hop)
bt = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
rms = librosa.feature.rms(y=y, hop_length=hop)[0]
t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
db = librosa.amplitude_to_db(rms, ref=np.max)
# strong hits: onset peaks well above local median
pk = librosa.util.peak_pick(onset, pre_max=6, post_max=6, pre_avg=20, post_avg=20, delta=np.percentile(onset, 85) * .35, wait=8)
hits = librosa.frames_to_time(pk, sr=sr, hop_length=hop)
# section boundaries via novelty on chroma+mfcc
feat = np.vstack([librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop), librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)])
bounds = librosa.segment.agglomerative(feat, 14)
bt_bounds = librosa.frames_to_time(bounds, sr=sr, hop_length=hop)
dur = len(y) / sr
cands = []
for b in bt_bounds:
    # snap to nearest beat
    if len(bt) == 0: break
    s = float(bt[np.argmin(abs(bt - b))])
    if s + win > dur - 2 or s < 3: continue
    m = (t >= s) & (t < s + win)
    loud = float(np.mean(db[m])); dens = float(np.mean(onset[(t >= s) & (t < s + win)]))
    cands.append({"start": round(s, 3), "loud_db": round(loud, 2), "onset": round(dens, 3)})
for c in cands:
    c["score"] = round(c["loud_db"] / 3 + c["onset"], 3)
cands.sort(key=lambda c: -c["score"])
out = {"tempo": float(np.atleast_1d(tempo)[0]), "dur": dur, "beats": [round(float(x), 3) for x in bt],
       "hits": [round(float(x), 3) for x in hits], "sections": [round(float(x), 2) for x in bt_bounds], "windows": cands[:6]}
json.dump(out, open(path.rsplit(".", 1)[0] + ".json", "w"))
print(f"{path}: tempo {out['tempo']:.1f}  dur {dur:.1f}s  beats {len(bt)}  hits {len(hits)}")
print("sections:", out["sections"])
for c in out["windows"]: print("  window", c)
