#!/usr/bin/env python3
"""Find where a short clip (iTunes 30 s preview = Apple's pick of the recognisable hook,
or a trending reel's audio) sits inside the full song.
Usage: hookfind.py full.wav clip.wav  -> prints offset seconds and match score"""
import sys, numpy as np, librosa
sr, hop = 11025, 512
a, _ = librosa.load(sys.argv[1], sr=sr); b, _ = librosa.load(sys.argv[2], sr=sr)
def feat(x):
    m = np.log1p(librosa.feature.melspectrogram(y=x, sr=sr, hop_length=hop, n_mels=48))
    m = (m - m.mean(0)) ; return m / (np.linalg.norm(m, axis=0) + 1e-9)
A, B = feat(a), feat(b)
B = B[:, 20:-20]; L = B.shape[1]
S = B.T @ A                      # (L, nA) frame similarities
n = A.shape[1] - L
score = np.array([np.trace(S[:, k:k + L]) / L for k in range(n)])
k = int(np.argmax(score)); off = (k - 20) * hop / sr
print(f"{off:.2f} score={score[k]:.3f} next_best={np.sort(score)[-int(5*sr/hop)]:.3f}")
