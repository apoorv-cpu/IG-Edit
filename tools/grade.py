"""House-look region grade (cloud port of regrade.py's idea).

Skin, garment and background are graded separately with MediaPipe selfie-multiclass masks,
smoothed over time:
  background : raw camera picture, untouched
  skin       : pulled toward R/G 1.50, B/G 0.78 (natural, golden)
  garment    : own hue, richer saturation (strength per colour); vivid fabric (>0.75 sat) left alone;
               optional `lift` (opens dark silk) and `clar` (local contrast on fabric)
  exposure   : gentle per-shot tone curve toward p50 luma ~155, real blacks, no blown whites
Finish (denoise, sharpen, 4K upscale) happens in ffmpeg, see render.py.
"""
import os, numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from reelkit import MODELS

SKIN_RG, SKIN_BG = 1.50, 0.78


class Segmenter:
    """Per-frame soft masks (skin, garment, person) at frame resolution, EMA-smoothed in time."""
    def __init__(self, ema=0.55):
        self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "selfie_multiclass_256x256.tflite")),
            output_confidence_masks=True))
        self.ema, self.prev = ema, None

    def reset(self):
        self.prev = None

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (288, 512), interpolation=cv2.INTER_AREA)
        r = self.seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)))
        c = [np.squeeze(m.numpy_view()) for m in r.confidence_masks]  # 0 bg,1 hair,2 body-skin,3 face-skin,4 clothes,5 others
        m = np.stack([c[2] + c[3], c[4], 1.0 - c[0]], -1).astype(np.float32)
        if self.prev is not None:
            m = self.ema * self.prev + (1 - self.ema) * m
        self.prev = m
        m = cv2.GaussianBlur(cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 3)
        return m[..., 0], m[..., 1], m[..., 2]


def _lum(x):
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


class ShotGrade:
    """Solve once per shot from sample frames, then apply to every frame."""
    def __init__(self, samples, seg, garment_strength=0.4, garment_sat_cap=0.75, lift=0.0, clar=0.0,
                 skin_strength=0.7, target_p50=155.0, extra_sat_bg=0.0):
        self.gs, self.cap, self.lift, self.clar = garment_strength, garment_sat_cap, lift, clar
        self.extra_sat_bg = extra_sat_bg
        rg, bg, sat, lum = [], [], [], []
        for f in samples:
            seg.reset()
            sk, gm, _ = seg(f)
            x = f.astype(np.float32)
            w = sk > 0.6
            if w.sum() > 500:
                R, G, B = (x[..., i][w].mean() for i in range(3))
                rg.append(R / max(G, 1)); bg.append(B / max(G, 1))
            gw = gm > 0.6
            if gw.sum() > 500:
                hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
                sat.append(np.median(hsv[..., 1][gw]) / 255)
            lum.append(np.percentile(_lum(x), 50))
        seg.reset()
        # skin gains (relative to G), limited to +-12 %
        if rg:
            r, b = np.median(rg), np.median(bg)
            gr = np.clip(SKIN_RG / r, 0.88, 1.12) ** skin_strength
            gb = np.clip(SKIN_BG / b, 0.88, 1.12) ** skin_strength
        else:
            gr = gb = 1.0
        self.skin_gain = np.array([gr, 1.0, gb], np.float32)
        self.g_sat = float(np.median(sat)) if sat else 0.5
        p50 = float(np.median(lum)) if lum else target_p50
        # exposure: gamma so p50 lands near target, capped (no milky lift)
        g = np.log(np.clip(target_p50, 1, 254) / 255) / np.log(np.clip(p50, 1, 254) / 255)
        self.gamma = float(np.clip(g, 0.85, 1.15))
        lut = (np.linspace(0, 1, 256) ** self.gamma) * 255
        # soft shoulder: keep whites from blowing
        lut = np.where(lut > 235, 235 + (lut - 235) * 0.6, lut)
        self.lut = np.clip(lut, 0, 255).astype(np.uint8)

    def __call__(self, rgb, seg):
        sk, gm, person = seg(rgb)
        x = cv2.LUT(rgb, self.lut).astype(np.float32)
        # skin
        if not np.allclose(self.skin_gain, 1):
            xs = np.clip(x * self.skin_gain, 0, 255)
            x = x + (xs - x) * sk[..., None]
        # garment saturation / lift / clarity
        if self.gs > 0 and self.g_sat < self.cap:
            hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            s = hsv[..., 1] / 255
            boost = 1 + self.gs * 0.55 * (1 - s)          # less push on already-rich pixels
            s2 = np.clip(s * boost, 0, 0.92)
            if self.lift:
                v = hsv[..., 2] / 255
                hsv[..., 2] = np.clip(v + self.lift * 0.25 * (1 - v) * (v < 0.5) * (0.5 - v) * 2, 0, 1) * 255
            hsv[..., 1] = s2 * 255
            xg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
            gmask = np.clip(gm - sk, 0, 1)
            x = x + (xg - x) * gmask[..., None]
        if self.clar:
            blur = cv2.GaussianBlur(x, (0, 0), 12)
            xc = x + self.clar * (x - blur)
            x = x + (xc - x) * gm[..., None]
        if self.extra_sat_bg:
            hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * (1 + self.extra_sat_bg), 0, 255)
            xb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
            x = x + (xb - x) * (1 - person)[..., None]
        return np.clip(x, 0, 255).astype(np.uint8)
