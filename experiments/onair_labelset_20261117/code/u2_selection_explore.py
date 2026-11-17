"""Exploration on development races: which boxes should a leading-group statement be made about?

Selection 'all': the front K boxes among every rider box (current rule).
Selection 'sure': the front K boxes among the boxes whose answer is sure.
usage: python u2_selection_explore.py dev
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "stripset"
split = sys.argv[1]
frames = [m for m in json.loads((S / "manifest.json").read_text())["frames"] if m["split"] == split]
TAGS = [None, "m05", "m10", "m15"]


def riders_of(frame, tag):
    p = (S / "out" / "D0" / f"{frame}.json") if tag is None else (S / "out_past" / "D0" / f"{frame}_{tag}.json")
    return json.loads(p.read_text())["riders"]


cache = {m["frame"]: [riders_of(m["frame"], t) for t in TAGS] for m in frames}


def said(frame, sel, W, K, mth):
    v = Counter()
    for riders in cache[frame][:W]:
        pool = sorted(riders, key=lambda r: -r["box"][3])
        if sel == "sure":
            pool = [r for r in pool if r["state"] == "sure"]
        for n in {r["no"] for r in pool[:K] if r["state"] == "sure"}:
            v[n] += 1
    return [n for n, c in v.items() if c >= mth]


print(f"split {split}: {len(frames)} frames")
print("selection W K m   statements   A (in top3)   in top5   B (frames with a correct statement)")
for sel in ("all", "sure"):
    for W, mth in ((1, 1), (3, 2), (4, 2), (4, 3)):
        for K in (1, 2, 3):
            st = ok = ok5 = fr = 0
            for m in frames:
                B = m["strip"]
                s = said(m["frame"], sel, W, K, mth)
                st += len(s)
                ok += sum(x in B[:3] for x in s)
                ok5 += sum(x in B for x in s)
                fr += any(x in B[:3] for x in s)
            print(f"{sel:5s}     {W} {K} {mth}   {st:4d}         {ok / st if st else 0:.3f}         {ok5 / st if st else 0:.3f}     {fr / len(frames):.3f}")
