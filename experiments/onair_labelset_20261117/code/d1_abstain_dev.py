"""Development analysis for an abstain rule. Uses ONLY batch 1 and batch 2A (both already scored).

Batch 2B is not touched. For every rider we rebuild the combined+assigned answer exactly as in
b2_evaluate.py (race-excluded training on batch 1) and record signals that could tell a right
answer from a wrong one. Writes results/abstain_dev_rows.json and prints summaries.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
W = 1.0


def usable(path):
    return [r for r in json.loads(path.read_text()) if r["human_number"] and not r["duplicate_of"]]


b1 = usable(ROOT / "results" / "evidence.json")
b2 = usable(ROOT / "batch2" / "results" / "evidence.json")
train_all = [r for r in b1 if r["R0_27"]["features"] is not None]
races = sorted({r["race"] for r in b1} | {r["race"] for r in b2})


def fit(rows):
    X = np.array([r["R0_27"]["features"] for r in rows])
    y = np.array([r["human_number"] for r in rows])
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)


models = {race: fit([r for r in train_all if r["race"] != race]) for race in races}


def proba(r):
    sc, lr = models[r["race"]]
    p = np.zeros(11)
    f = r["R0_27"]["features"]
    if f is not None:
        p[np.array(lr.classes_) - 1] = lr.predict_proba(sc.transform(np.array([f])))[0]
    return p


out = []
for batch, rows in (("1", b1), ("2A", b2)):
    frames = defaultdict(list)
    for r in rows:
        frames[r["frame"]].append(r)
    for f, items in frames.items():
        P = np.array([proba(r) for r in items])
        S = P.copy()
        for i, r in enumerate(items):
            q = r["R0_35"]["vlm"]["number"]
            if q:
                S[i, q - 1] += W
        ri, cj = linear_sum_assignment(-S)
        for i, j in zip(ri, cj):
            r = items[i]
            q = r["R0_35"]["vlm"]["number"]
            others = np.delete(S[i], j)
            out.append({
                "id": r["id"], "batch": batch, "race": r["race"], "frame": f, "human": r["human_number"],
                "assigned": int(j) + 1, "correct": bool(int(j) + 1 == r["human_number"]),
                "qwen": q, "qwen_state": "agree" if q == j + 1 else ("abstain" if q == 0 else "overridden"),
                "p_lr_assigned": float(P[i, j]), "p_lr_top": float(P[i].max()), "lr_top": int(P[i].argmax()) + 1,
                "margin": float(S[i, j] - others.max()), "height_px": r["height_px"], "riders_in_frame": len(items),
                "hsv_number": r["R0_27"]["hsv"]["number"],
            })
(ROOT / "results" / "abstain_dev_rows.json").write_text(json.dumps(out, indent=1))


def pc(sel):
    n = len(sel)
    c = sum(r["correct"] for r in sel)
    return f"{c:3d}/{n:3d} ({(c / n if n else 0):.2f})"


for batch in ("1", "2A", "both"):
    rows = [r for r in out if batch == "both" or r["batch"] == batch]
    print(f"\n=== batch {batch}: all {pc(rows)}")
    for st in ("agree", "abstain", "overridden"):
        print(f"  qwen {st:10s} {pc([r for r in rows if r['qwen_state'] == st])}")
    print("  by assigned number: " + "  ".join(f"{n}:{pc([r for r in rows if r['assigned'] == n])}" for n in range(1, 12)))
    for st in ("agree", "abstain"):
        sel = [r for r in rows if r["qwen_state"] == st]
        print(f"  qwen {st}: precision when p_lr_assigned >= t")
        for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            print(f"     t={t:.1f}  {pc([r for r in sel if r['p_lr_assigned'] >= t])}")
    sel = [r for r in rows if r["qwen_state"] == "agree"]
    print("  qwen agree, by assigned number: " + "  ".join(f"{n}:{pc([r for r in sel if r['assigned'] == n])}" for n in range(1, 12)))
    sel = [r for r in rows if r["qwen_state"] == "abstain"]
    print("  qwen abstain, by assigned number: " + "  ".join(f"{n}:{pc([r for r in sel if r['assigned'] == n])}" for n in range(1, 12)))
    print("  qwen abstain, lr_top == assigned: " + pc([r for r in sel if r["lr_top"] == r["assigned"]]) + " | hsv agrees: " + pc([r for r in sel if r["hsv_number"] == r["assigned"]]))
    print("  by height: " + "  ".join(f"{lo}-{hi}:{pc([r for r in rows if lo <= r['height_px'] < hi])}" for lo, hi in ((70, 100), (100, 150), (150, 1000))))
