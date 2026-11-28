#!/usr/bin/env python3
"""Plan (JSON) -> finished reel: 1080x1920 + 2160x3840, region-graded, hard cuts on the music.

Usage: render.py plan.json out_dir [--sheet]   (--sheet: only a before/after grade sheet)

Plan keys:
  name, clips_dir, song (wav), song_json (from songscan.py), song_start (s in the song)
  shots: [{clip, in, speed (1.0|0.5), len (approx s), move, note}]
  snap: "hits" | "beats"   cut points snap to the nearest strong hit/beat (within 0.25 s)
  grade: {strength, sat_cap, lift, clar, skin, p50}
  end_fade: seconds of audio fade at the end
Rules enforced: speed only 1.0 or 0.5; every shot must open in motion (start check);
no source second used twice; frame rate/count verified after render.
"""
import sys, os, json, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from reelkit import read_frames, probe, Writer, FFMPEG, FFPROBE, OUT_FPS
from grade import Segmenter, ShotGrade


def snap_cuts(plan, song):
    pts = np.array(song["hits"] if plan.get("snap", "beats") == "hits" else song["beats"])
    beats = np.array(song["beats"])
    t0 = plan["song_start"]; t = t0; cuts = [t0]
    for s in plan["shots"]:
        want = t + s["len"]
        cand = pts[(pts > t + 0.6) & (abs(pts - want) < 0.35)]
        if not len(cand):
            cand = beats[(beats > t + 0.6) & (abs(beats - want) < 0.5)]
        t = float(cand[np.argmin(abs(cand - want))]) if len(cand) else want
        cuts.append(t)
    return cuts


def check_overlap(plan, cuts):
    used = {}
    for i, s in enumerate(plan["shots"]):
        a = s["in"]; b = a + (cuts[i + 1] - cuts[i]) * s["speed"]
        for (c, d, j) in used.get(s["clip"], []):
            if a < d - 0.05 and c < b - 0.05:
                raise SystemExit(f"shot {i} reuses {s['clip']} {max(a,c):.2f}-{min(b,d):.2f}s already used by shot {j}")
        used.setdefault(s["clip"], []).append((a, b, i))


def shot_frames(path, t_in, dur, speed):
    n = int(round(dur * OUT_FPS))
    if speed == 1.0:
        it = sharp_pairs(read_frames(path, t_in, dur + 0.2, fps=None))   # 59.94 -> 30: keep the sharper of each pair
    elif speed == 0.5:
        it = read_frames(path, t_in, dur * 0.5 + 0.2, fps=None)   # native 59.94 frames played at 30
    else:
        raise SystemExit("speed must be 1.0 or 0.5")
    out = []
    for f in it:
        out.append(f)
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit(f"{os.path.basename(path)} @ {t_in:.2f}: only {len(out)}/{n} frames (clip too short)")
    return out


def _lap(f):
    g = cv2.cvtColor(cv2.resize(f, (270, 480)), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(g, cv2.CV_32F).var()


def sharp_pairs(it):
    buf = []
    for f in it:
        buf.append(f)
        if len(buf) == 2:
            yield buf[0] if _lap(buf[0]) >= _lap(buf[1]) else buf[1]
            buf = []


def start_motion(frames, k=8):
    g = [cv2.cvtColor(cv2.resize(f, (270, 480)), cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames[:k + 1]]
    return float(np.mean([np.mean(np.abs(g[i + 1] - g[i])) for i in range(k)]) / 2.55)


def main():
    plan = json.load(open(sys.argv[1])); out_dir = sys.argv[2]
    sheet_only = "--sheet" in sys.argv
    os.makedirs(out_dir, exist_ok=True)
    song = json.load(open(plan["song_json"]))
    cuts = snap_cuts(plan, song)
    check_overlap(plan, cuts)
    total = cuts[-1] - cuts[0]
    gp = plan.get("grade", {})
    seg = Segmenter()
    name = plan["name"]
    tmp = os.path.join(out_dir, f".{name}_graded.mp4")
    W = None if sheet_only else Writer(tmp, 1080, 1920, crf=10, preset="medium")
    report, sheet, thumbs = [], [], []
    for i, s in enumerate(plan["shots"]):
        dur = cuts[i + 1] - cuts[i]
        path = os.path.join(plan["clips_dir"], s["clip"])
        frames = shot_frames(path, s["in"], dur, s["speed"])
        mot = start_motion(frames)
        samples = [frames[0], frames[len(frames) // 2], frames[-1]]
        g = ShotGrade(samples, seg, garment_strength=gp.get("strength", 0.4), garment_sat_cap=gp.get("sat_cap", 0.75),
                      lift=gp.get("lift", 0.0), clar=gp.get("clar", 0.0), skin_strength=gp.get("skin", 0.7),
                      target_p50=gp.get("p50", 155),
                      gamma_range=tuple(gp.get("gamma_range", (0.85, 1.15))), bright=gp.get("bright", 0.0))
        seg.reset()
        flag = "" if mot >= plan.get("min_start_motion", 0.6) else "  <-- STILL START"
        report.append(f"shot {i:2d} {s['clip']} in {s['in']:.2f} x{s['speed']} {dur:.2f}s  start-motion {mot:.2f}"
                      f"  gamma {g.gamma:.2f} skin {g.skin_gain.round(3).tolist()} gsat {g.g_sat:.2f}  {s.get('move','')}{flag}")
        print(report[-1], flush=True)
        if sheet_only:
            mid = frames[len(frames) // 2]
            sheet.append(np.vstack([cv2.resize(mid, (270, 480)), cv2.resize(g(mid, seg), (270, 480))]))
            seg.reset(); continue
        for k, f in enumerate(frames):
            o = g(f, seg); W.write(o)
            if k in (0, 4, len(frames) // 2, len(frames) - 1):
                thumbs.append(cv2.resize(o, (180, 320)))
        seg.reset()
    if sheet_only:
        img = np.hstack(sheet)[:, :, ::-1]
        cv2.imwrite(os.path.join(out_dir, f"{name}_grade_sheet.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return
    W.close()
    rows = [np.hstack(thumbs[i:i + 4]) for i in range(0, len(thumbs), 4)]
    rows = [np.hstack(rows[i:i + 3]) if len(rows[i:i + 3]) == 3 else np.hstack(rows[i:i + 3] + [np.zeros_like(rows[0])] * (3 - len(rows[i:i + 3]))) for i in range(0, len(rows), 3)]
    cv2.imwrite(os.path.join(out_dir, f"{name}_shots.jpg"), np.vstack(rows)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    # audio: song section, gentle fade at the end, -14 LUFS
    aud = os.path.join(out_dir, f".{name}_audio.m4a")
    fade = plan.get("end_fade", 1.5)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{cuts[0]:.3f}", "-t", f"{total:.3f}", "-i", plan["song"],
                    "-af", f"afade=t=in:d=0.02,afade=t=out:st={total - fade:.3f}:d={fade},loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-ar", "48000", "-c:a", "aac", "-b:a", "256k", aud], check=True)
    base = os.path.join(out_dir, name)
    dn = plan.get("denoise", "hqdn3d=1.0:0.8:2.2:1.8")   # set temporal parts to 0 to avoid motion smear
    f1080 = f"{base}_1080p.mp4"; f4k = f"{base}_4K.mp4"
    only4k = plan.get("only_4k", False)
    if not only4k: subprocess.run([FFMPEG, "-v", "error", "-y", "-i", tmp, "-i", aud, "-map", "0:v", "-map", "1:a",
                    "-vf", f"{dn},unsharp=5:5:0.35",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30",
                    "-c:a", "copy", "-shortest", "-movflags", "+faststart", f1080], check=True)
    hook = plan.get("hook")
    v4 = f"{dn},scale=2160:3840:flags=lanczos,unsharp=5:5:0.40"
    if hook:
        from hooktext import make
        png = os.path.join(out_dir, f".{name}_hook.png"); make(hook["text"], png)
        hd = hook.get("dur", 3.0)
        vargs = ["-loop", "1", "-t", f"{hd}", "-i", png, "-filter_complex",
                 f"[0:v]{v4}[b];[2:v]format=rgba,fade=t=out:st={hd - 0.3}:d=0.3:alpha=1[t];"
                 f"[b][t]overlay=0:0:eof_action=pass[v]", "-map", "[v]"]
    else:
        vargs = ["-vf", v4, "-map", "0:v"]
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", tmp, "-i", aud] + vargs + ["-map", "1:a",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
                    "-profile:v", "high", "-level", "5.1",
                    "-c:a", "copy", "-shortest", "-movflags", "+faststart", f4k], check=True)
    os.remove(tmp); os.remove(aud)
    if hook: os.remove(png)
    # verify
    for f in ((f4k,) if only4k else (f1080, f4k)):
        r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                            "stream=width,height,r_frame_rate,nb_read_frames", "-of", "csv=p=0", f],
                           capture_output=True, text=True).stdout.strip()
        report.append(f"{os.path.basename(f)}: {r}  (expected {int(round(total * OUT_FPS))} frames)")
    report.append(f"cuts (song time): {[round(c, 3) for c in cuts]}")
    report.append(f"length {total:.2f}s, song start {cuts[0]:.2f}s")
    open(f"{base}_report.txt", "w").write("\n".join(report) + "\n")
    print("\n".join(report[-4:]))


if __name__ == "__main__":
    main()
