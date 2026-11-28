#!/usr/bin/env python3
"""Hand-picked shot list -> plan. spec2plan.py draft_plan.json "clip:in:L, clip:in:L, ..." (L = source seconds; default 0.5x, --rt = 1.0x, or clip:in:L:speed)"""
import sys, json
p = json.load(open(sys.argv[1])); shots = []
for tok in sys.argv[2].split(","):
    c, a, L, *sp = tok.strip().split(":")
    sp = float(sp[0]) if sp else (1.0 if "--rt" in sys.argv else 0.5)
    shots.append(dict(clip=f"DSC_{c}.MOV", **{"in": float(a)}, speed=sp, len=round(float(L) / sp, 2), move="picked"))
p["shots"] = shots; json.dump(p, open(sys.argv[1], "w"), indent=1)
print(len(shots), "shots", sum(s["len"] for s in shots), "s")
