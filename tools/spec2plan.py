#!/usr/bin/env python3
"""Hand-picked shot list -> plan. spec2plan.py draft_plan.json "clip:in:L, clip:in:L, ..." (L = source seconds, 0.5x)"""
import sys, json
p = json.load(open(sys.argv[1])); shots = []
for tok in sys.argv[2].split(","):
    c, a, L = tok.strip().split(":")
    shots.append(dict(clip=f"DSC_{c}.MOV", **{"in": float(a)}, speed=0.5, len=round(2 * float(L), 2), move="picked"))
p["shots"] = shots; json.dump(p, open(sys.argv[1], "w"), indent=1)
print(len(shots), "shots", sum(s["len"] for s in shots), "s")
