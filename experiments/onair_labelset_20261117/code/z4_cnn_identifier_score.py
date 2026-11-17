"""Identifier variants with the fine-tuned cap classifier, scored against the strip. See cnn/DESIGN_IDENTIFIER_KO.md.

usage: python z4_cnn_identifier_score.py dev
       python z4_cnn_identifier_score.py test VARIANT W K m
"""
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "stripset"
split = sys.argv[1]
frames = [m for m in json.loads((S / "manifest.json").read_text())["frames"] if m["split"] == split]
TAGS = [None, "m05", "m10"]
CONF = 0.8


def load(frame, tag):
    name = frame if tag is None else f"{frame}_{tag}"
    riders = json.loads(((S / "out" / "D0") if tag is None else (S / "out_past" / "D0")).joinpath(f"{name}.json").read_text())["riders"]
    probs = json.loads((S / "cnn_probs" / f"{name}.json").read_text())["probs"]
    assert len(riders) == len(probs)
    return riders, np.array(probs) if probs else np.zeros((0, 11))


def label(riders, P, variant):
    """Returns [(bottom_y, number, sure)] for one frame."""
    if variant == "BASE":
        return [(r["box"][3], r["no"], r["state"] == "sure") for r in riders]
    if not riders:
        return []
    S_ = P.copy()
    q = [r["qwen_number"] for r in riders]
    if variant in ("C2", "C3"):
        for i, n in enumerate(q):
            if n:
                S_[i, n - 1] += 1.0
    ri, cj = linear_sum_assignment(-S_)
    assigned = {int(i): int(j) for i, j in zip(ri, cj)}
    out = []
    for i, r in enumerate(riders):
        j = assigned.get(i)
        if j is None:
            out.append((r["box"][3], 0, False))
            continue
        no, conf = j + 1, P[i, j] >= CONF
        if variant == "C1":
            sure = conf
        elif variant == "C2":
            sure = (q[i] == no) or conf
        else:
            sure = conf and (q[i] == no or q[i] == 0)
        out.append((r["box"][3], no, bool(sure)))
    return out


cache = {}
for m in frames:
    cache[m["frame"]] = [load(m["frame"], t) for t in TAGS]


def statements(frame, variant, W, K, mth):
    votes = Counter()
    for riders, P in cache[frame][:W]:
        lab = sorted(label(riders, P, variant), key=lambda x: -x[0])[:K]
        for n in {n for _, n, s in lab if s and n}:
            votes[n] += 1
    return [n for n, v in votes.items() if v >= mth]


def evaluate(variant, W, K, mth):
    st = ok = ok5 = 0
    good = {}
    for m in frames:
        B = m["strip"]
        said = statements(m["frame"], variant, W, K, mth)
        st += len(said)
        ok += sum(n in B[:3] for n in said)
        ok5 += sum(n in B for n in said)
        good[m["frame"]] = any(n in B[:3] for n in said)
    return {"variant": variant, "W": W, "K": K, "m": mth, "statements": st, "A": ok / st if st else 0.0, "A_top5": ok5 / st if st else 0.0,
            "B": sum(good.values()) / len(frames)}, good


fmt = lambda v: f"{v['variant']:4s} W{v['W']} K{v['K']} m{v['m']}  statements {v['statements']:4d}  A {v['A']:.3f} (top5 {v['A_top5']:.3f})  B {v['B']:.3f}"
base, base_good = evaluate("BASE", 3, 2, 2)
out = {"split": split, "frames": len(frames), "baseline": base}
print(f"split {split}: {len(frames)} frames\nbaseline: {fmt(base)}")
if split == "dev":
    configs = [(v, W, K, mth) for v in ("C1", "C2", "C3") for (W, mth) in ((1, 1), (3, 2)) for K in (1, 2, 3)]
    res = {c: evaluate(*c)[0] for c in configs}
    for c in configs:
        print(("* " if res[c]["A"] >= base["A"] else "  ") + fmt(res[c]))
    ok = [c for c in configs if res[c]["A"] >= base["A"]]
    out["configs"] = [res[c] for c in configs]
    if ok:
        best = max(ok, key=lambda c: (round(res[c]["B"], 6), round(res[c]["A"], 6), -("C1", "C2", "C3").index(c[0]), -c[1], -c[2]))
        out["chosen"] = list(best)
        print("chosen on dev:", best, "(* = A at least the baseline's)")
    else:
        out["chosen"] = None
        print("no configuration keeps A at the baseline's level on dev")
else:
    c = (sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    v, good = evaluate(*c)
    imp = sum(good[f] and not base_good[f] for f in good)
    wor = sum(base_good[f] and not good[f] for f in good)
    n, mm = imp + wor, min(imp, wor)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(mm + 1)) / 2 ** n) if n else 1.0
    better = bool(imp > wor and p < 0.05 and v["A"] >= base["A"] - 0.02)
    out["confirm"] = {"config": list(c), "result": v, "improved_frames": imp, "worsened_frames": wor, "sign_test_p": p, "better": better,
                      "meets_goal_A_90": bool(v["A"] >= 0.90), "meets_goal_B_80": bool(v["B"] >= 0.80)}
    print("chosen  :", fmt(v))
    print(f"frames: improved {imp} / worsened {wor}  p={p:.4f} | A change {v['A'] - base['A']:+.3f} | better: {better} | goal A>=0.90: {v['A'] >= 0.90}  goal B>=0.80: {v['B'] >= 0.80}")
(ROOT / "cnn" / f"score_identifier_{split}.json").write_text(json.dumps(out, indent=1))
