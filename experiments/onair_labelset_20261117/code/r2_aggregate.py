"""Multi-frame statements: number N is in the leading group if it is sure among the front K boxes in at least m of the last W frames.

usage: python r2_aggregate.py dev            -> all configs on development races, picks one by the fixed rule
       python r2_aggregate.py test W K m     -> compares that config with the single-frame baseline on held-out races
See multi/DESIGN_KO.md. No model calls. Writes multi/score_<split>.json.
"""
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "stripset"
split = sys.argv[1]
frames = [m for m in json.loads((S / "manifest.json").read_text())["frames"] if m["split"] == split]
TAGS = [None, "m05", "m10", "m15"]  # most recent first


def riders_of(frame, tag):
    p = (S / "out" / "D0" / f"{frame}.json") if tag is None else (S / "out_past" / "D0" / f"{frame}_{tag}.json")
    return json.loads(p.read_text())["riders"]


cache = {m["frame"]: [riders_of(m["frame"], t) for t in TAGS] for m in frames}


def statements(frame, W, K, m):
    votes = Counter()
    for riders in cache[frame][:W]:
        front = sorted(riders, key=lambda r: -r["box"][3])[:K]
        for n in {r["no"] for r in front if r["state"] == "sure"}:
            votes[n] += 1
    return [n for n, v in votes.items() if v >= m]


def evaluate(W, K, m):
    st = ok3 = ok5 = any_ = 0
    good = {}
    for f in frames:
        B = f["strip"]
        said = statements(f["frame"], W, K, m)
        st += len(said)
        ok3 += sum(n in B[:3] for n in said)
        ok5 += sum(n in B for n in said)
        any_ += bool(said)
        good[f["frame"]] = any(n in B[:3] for n in said)
    n = len(frames)
    return {"W": W, "K": K, "m": m, "statements": st, "in_top3": ok3, "A": ok3 / st if st else 0.0, "A_top5": ok5 / st if st else 0.0,
            "frames_with_statement": any_ / n, "B": sum(good.values()) / n}, good


configs = [(1, 1, 1)] + [(W, K, m) for W in (2, 3, 4) for K in (1, 2, 3) for m in range(1, W + 1)]
res = {c: evaluate(*c) for c in configs}
base = res[(1, 1, 1)][0]
out = {"split": split, "frames": len(frames), "baseline": base, "configs": [res[c][0] for c in configs]}
fmt = lambda v: f"W{v['W']} K{v['K']} m{v['m']}  statements {v['statements']:4d}  A {v['A']:.3f} (top5 {v['A_top5']:.3f})  frames with statement {v['frames_with_statement']:.3f}  B {v['B']:.3f}"
print(f"split {split}: {len(frames)} frames")
if split == "dev":
    for c in configs:
        print(("* " if res[c][0]["A"] >= base["A"] else "  ") + fmt(res[c][0]))
    ok = [c for c in configs if res[c][0]["A"] >= base["A"]]
    best = max(ok, key=lambda c: (round(res[c][0]["B"], 6), round(res[c][0]["A"], 6), -c[0], -c[1]))
    out["chosen"] = list(best)
    print("baseline:", fmt(base))
    print("chosen on dev:", best, "(* = A at least the baseline's)")
else:
    c = tuple(int(x) for x in sys.argv[2:5])
    v, good = res[c]
    bgood = res[(1, 1, 1)][1]
    imp = sum(good[f] and not bgood[f] for f in good)
    wor = sum(bgood[f] and not good[f] for f in good)
    n, mm = imp + wor, min(imp, wor)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(mm + 1)) / 2 ** n) if n else 1.0
    better = bool(imp > wor and p < 0.05 and v["A"] >= base["A"] - 0.02)
    out["confirm"] = {"config": list(c), "result": v, "improved_frames": imp, "worsened_frames": wor, "sign_test_p": p, "better": better}
    print("baseline:", fmt(base))
    print("chosen  :", fmt(v))
    print(f"frames: improved {imp} / worsened {wor}  p={p:.4f} | A change {v['A'] - base['A']:+.3f} | better: {better}")
(ROOT / "multi").mkdir(exist_ok=True)
(ROOT / "multi" / f"score_{split}.json").write_text(json.dumps(out, indent=1))
