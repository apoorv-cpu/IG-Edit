# IG-Edit: Daabu reel pipeline (cloud)

Turns model clips shared in a Google Drive folder into 4K Instagram reels, entirely in a cloud
session, so the laptop can stay closed.

## Flow for each Drive link

1. **Fetch.** List the folder with the Google Drive connector, write `manifest.tsv` (`subfolder<TAB>file_id<TAB>name`), then
   `python3 tools/fetch_drive.py manifest.tsv clips/`. The folder must be shared as "anyone with the link".
2. **Review footage.** `tools/overview.py clips/N sheet.jpg` makes one row per clip at 1 fps.
   `tools/analyze.py clips/N` writes motion, sharpness and face curves, and `tools/moments.py clips/N` prints them.
3. **Music.** Pick a trending song for the category and mood. Download it from JioSaavn with yt-dlp, then run
   `tools/songscan.py song.wav` to get beats, strong hits, sections and candidate 27 s windows.
4. **Plan.** Write `plans/<reel>.json`: the shots (clip, in-point, speed 1.0 or 0.5, approx length, move) and the song start.
5. **Grade check.** `tools/render.py plan.json out/ --sheet` makes a before/after grade sheet.
6. **Render.** `tools/render.py plan.json out/` produces `_1080p.mp4`, `_4K.mp4` (2160x3840, 30 fps, -14 LUFS) and `_report.txt`.
   The report has the start-motion check per shot, frame counts and the cut list.
7. **Deliver.** Upload to Drive. Add the caption and audio note, then log the reel in the tracker (`tracker/`).

## House rules (from Daabu learnings)

- 24 to 30 s. Open on the most striking moving shot with the face visible at frame 0. No fade-in.
- Every shot opens already in motion. No still or freeze frames. Never reuse the same source seconds.
- Speed is 1.0x or 0.5x only (60 fps source). Slow motion is for fabric and twirls, not faces.
- One move type per shot: at most one twirl and one hair flip per reel.
- Skip assistant hands, phones, talking, looking down to adjust, soft focus, and foreground people.
- Grade: background stays raw. Skin is natural and golden (R/G 1.50, B/G 0.78). The garment keeps its own hue with richer saturation.
  Fabric already above 0.75 saturation is left alone. Light denoise and light sharpen only.
- Captions: a hook line, what the outfit is, one CTA with a comment keyword, the store address, and five hashtags.

## Requirements

`pip install -r requirements.txt` and `apt-get install libegl1 libgles2` (MediaPipe), plus ffmpeg.
Models are in `tools/models/` (MediaPipe selfie multiclass, BlazeFace).
Note: librosa's numba beat tracker segfaults in the cloud image, so `songscan.py` has its own numpy DP tracker.
