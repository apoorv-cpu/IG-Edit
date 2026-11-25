#!/usr/bin/env python3
"""Download every video in a link-shared Drive folder tree.

Usage: fetch_drive.py manifest.tsv out_dir
manifest.tsv lines: <subfolder>\t<file_id>\t<file_name>
(The manifest comes from the Google Drive connector listing; the cloud
workspace downloads through drive.usercontent.google.com, which serves
link-shared files without sign-in.)
"""
import subprocess, sys, os, concurrent.futures as cf

def get(row, out):
    sub, fid, name = row
    d = os.path.join(out, sub); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    if os.path.exists(p) and os.path.getsize(p) > 1_000_000:
        return p, "cached"
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    for _ in range(4):
        r = subprocess.run(["curl", "-sfL", "-m", "900", "-o", p + ".part", url])
        if r.returncode == 0:
            os.replace(p + ".part", p); return p, "ok"
    return p, "FAILED"

if __name__ == "__main__":
    rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
    with cf.ThreadPoolExecutor(6) as ex:
        for p, s in ex.map(lambda r: get(r, sys.argv[2]), rows):
            print(s, p, flush=True)
