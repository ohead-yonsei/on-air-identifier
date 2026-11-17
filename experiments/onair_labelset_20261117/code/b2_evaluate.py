"""Batch-2 confirmation test. Fixed before any batch-2 frame was extracted. See batch2/DESIGN_KO.md.

Train: batch-1 riders only (results/evidence.json). Test: batch-2 riders (batch2/results/evidence.json).
Writes batch2/results/summary.json and batch2/results/rows.csv.
"""
import csv
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
B2 = ROOT / "batch2"
W = 1.0
NUMS = list(range(1, 12))


def usable(path):
    return [r for r in json.loads(path.read_text()) if r["human_number"] and not r["duplicate_of"]]


train_all = [r for r in usable(ROOT / "results" / "evidence.json") if r["R0_27"]["features"] is not None]
test = usable(B2 / "results" / "evidence.json")
assert all(r["human_number"] in NUMS for r in test)
races = sorted({r["race"] for r in test})
frames = defaultdict(list)
for r in test:
    frames[r["frame"]].append(r)
pred = {r["id"]: {} for r in test}


def fit(rows):
    X = np.array([r["R0_27"]["features"] for r in rows])
    y = np.array([r["human_number"] for r in rows])
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)


def proba(model, r):
    sc, lr = model
    p = np.zeros(11)
    f = r["R0_27"]["features"]
    if f is not None:
        p[np.array(lr.classes_) - 1] = lr.predict_proba(sc.transform(np.array([f])))[0]
    return p


def assign(tag, S):
    for f, items in frames.items():
        M = np.array([S[r["id"]] for r in items])
        ri, cj = linear_sum_assignment(-M)
        for i, j in zip(ri, cj):
            pred[items[i]["id"]][tag] = (int(j) + 1) if M[i, j] > 0 else 0


for r in test:
    pred[r["id"]]["Q"] = r["R0_35"]["vlm"]["number"]
    pred[r["id"]]["H"] = r["R0_27"]["hsv"]["number"]

model_all = fit(train_all)
models_loro = {race: fit([r for r in train_all if r["race"] != race]) for race in races}
for variant in ("loro", "all"):
    P, S = {}, {}
    for r in test:
        p = proba(models_loro[r["race"]] if variant == "loro" else model_all, r)
        s = p.copy()
        q = r["R0_35"]["vlm"]["number"]
        if q:
            s[q - 1] += W
        P[r["id"]], S[r["id"]] = p, s
        pred[r["id"]][f"L_I.{variant}"] = (int(np.argmax(p)) + 1) if p.max() > 0 else 0
        pred[r["id"]][f"C_I.{variant}"] = (int(np.argmax(s)) + 1) if s.max() > 0 else 0
    assign(f"L_A.{variant}", P)
    assign(f"C_A.{variant}", S)

CONDS = ["Q", "H"] + [f"{c}.{v}" for v in ("loro", "all") for c in ("L_I", "L_A", "C_I", "C_A")]


def verdict(p, t):
    return "abstain" if p == 0 else ("correct" if p == t else "wrong")


def tally(sel, c):
    k = Counter(verdict(pred[r["id"]][c], r["human_number"]) for r in sel)
    return {"n": len(sel), "correct": k["correct"], "wrong": k["wrong"], "abstain": k["abstain"]}


def paired(a, b):
    k = Counter()
    for r in test:
        ca, cb = pred[r["id"]][a] == r["human_number"], pred[r["id"]][b] == r["human_number"]
        k["improved" if cb and not ca else "worsened" if ca and not cb else "same"] += 1
    n, m = k["improved"] + k["worsened"], min(k["improved"], k["worsened"])
    p = min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0
    return {"improved": k["improved"], "worsened": k["worsened"], "same": k["same"], "sign_test_p": p}


primary = paired("Q", "C_A.loro")
summary = {
    "n_train_batch1": len(train_all), "n_test_batch2": len(test), "n_frames": len(frames), "W": W,
    "overall": {c: tally(test, c) for c in CONDS},
    "primary_Q_vs_C_A_loro": primary,
    "replicated": bool(primary["improved"] > primary["worsened"] and primary["sign_test_p"] < 0.05),
    "paired": {f"{a} -> {b}": paired(a, b) for a, b in [
        ("Q", "C_A.all"), ("Q", "C_I.loro"), ("C_I.loro", "C_A.loro"), ("Q", "L_A.loro"), ("H", "L_I.loro"), ("L_I.loro", "L_A.loro")]},
    "by_number": {c: {n: tally([r for r in test if r["human_number"] == n], c) for n in NUMS} for c in ("Q", "H", "L_A.loro", "C_A.loro", "C_A.all")},
    "by_race": {c: {race: tally([r for r in test if r["race"] == race], c) for race in races} for c in ("Q", "C_A.loro", "C_A.all")},
    "confusion": {c: {n: dict(Counter(pred[r["id"]][c] for r in test if r["human_number"] == n)) for n in NUMS} for c in ("Q", "C_A.loro")},
}
(B2 / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
with open(B2 / "results" / "rows.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "race", "frame", "height_px", "human_number"] + CONDS)
    for r in test:
        w.writerow([r["id"], r["race"], r["frame"], r["height_px"], r["human_number"]] + [pred[r["id"]][c] for c in CONDS])

f3 = lambda d: f"{d['correct']:3d}/{d['wrong']:3d}/{d['abstain']:3d}"
print(f"train (batch 1) {len(train_all)}   test (batch 2) {len(test)}   frames {len(frames)}")
for c in CONDS:
    print(f"  {c:10s} {f3(summary['overall'][c])}")
print("PRIMARY  Q -> C_A.loro:", primary, "| replicated:", summary["replicated"])
for k, v in summary["paired"].items():
    print(f"  {k:22s} {v['improved']:3d} / {v['worsened']:3d}  p={v['sign_test_p']:.4f}")
print("correct by number:")
print("  n          " + " ".join(f"{summary['by_number']['Q'][n]['n']:>4d}" for n in NUMS))
for c in summary["by_number"]:
    print(f"  {c:10s} " + " ".join(f"{summary['by_number'][c][n]['correct']:>4d}" for n in NUMS))
print("correct by race:")
print("  n          " + " ".join(f"{summary['by_race']['Q'][r]['n']:>4d}" for r in races))
for c in summary["by_race"]:
    print(f"  {c:10s} " + " ".join(f"{summary['by_race'][c][r]['correct']:>4d}" for r in races))
for c in summary["confusion"]:
    print("confusion", c)
    for n in NUMS:
        d = summary["confusion"][c][n]
        print(f"  true {n:2d}: " + ", ".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda x: -x[1])))
