"""Shared helpers for the Daabu reel pipeline (cloud version).

Footage: Nikon DSC_*.MOV, 1920x1080, 59.94 fps, shot sideways -> transpose=2.
Output: 1080x1920 (and 2160x3840) at 30 fps.
"""
import json, os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")
OUT_FPS = 30
ROTATE = os.environ.get("REEL_ROTATE", "transpose=2")


def probe(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
                        "-of", "json", path], capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": float(num) / float(den),
            "frames": int(s.get("nb_frames", 0)), "dur": float(s["duration"])}


def read_frames(path, start, dur, w=1080, h=1920, fps=None, rotate=ROTATE):
    """Yield RGB uint8 frames (h, w, 3) from [start, start+dur) of a clip, rotated upright.
    fps=None keeps every native frame (use for 0.5x slow motion)."""
    vf = [rotate] if rotate else []
    if fps:
        vf.append(f"fps={fps}")
    vf.append(f"scale={w}:{h}:flags=lanczos")
    cmd = [FFMPEG, "-v", "fatal", "-ss", f"{start:.3f}", "-i", path, "-t", f"{dur:.3f}",
           "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close(); p.wait()


class Writer:
    """Raw RGB frames -> H.264 file. Note -framerate (not -r) on the raw input."""
    def __init__(self, path, w, h, fps=OUT_FPS, crf=14, vf=None, preset="slow"):
        cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "-"]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-profile:v", "high", "-movflags", "+faststart", path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())

    def close(self):
        self.p.stdin.close(); self.p.wait()
        if self.p.returncode:
            raise RuntimeError("ffmpeg encode failed")
