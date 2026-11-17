"""Does more training data help? Fixed before running. See moredata/DESIGN_KO.md.

Leave-one-race-out over all 417 labelled riders. Compares training on batch 1 only ("small")
with training on all three batches ("large"), other races only in both cases.
Writes moredata/summary.json and moredata/rows.csv. No model calls.
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
OUT = ROOT / "moredata"
W, P_MIN = 1.0, 0.1
NUMS = list(range(1, 12))


def usable(path, batch):
    rows = [r for r in json.loads(path.read_text()) if r["human_number"] and not r["duplicate_of"]]
    for r in rows:
        r["batch"] = batch
    return rows


rows = (usable(ROOT / "results" / "evidence.json", "1") + usable(ROOT / "batch2" / "results" / "evidence.json", "2A")
        + usable(ROOT / "batch2b" / "results" / "evidence.json", "2B"))
assert len({(r["batch"], r["id"]) for r in rows}) == len(rows)
races = sorted({r["race"] for r in rows})
key = lambda r: f'{r["batch"]}:{r["id"]}'


def fit(sel):
    sel = [r for r in sel if r["R0_27"]["features"] is not None]
    X = np.array([r["R0_27"]["features"] for r in sel])
    y = np.array([r["human_number"] for r in sel])
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y), len(sel)


def proba(model, r):
    sc, lr, _ = model
    p = np.zeros(11)
    f = r["R0_27"]["features"]
    if f is not None:
        p[np.array(lr.classes_) - 1] = lr.predict_proba(sc.transform(np.array([f])))[0]
    return p


models = {"small": {}, "large": {}}
for race in races:
    models["small"][race] = fit([r for r in rows if r["race"] != race and r["batch"] == "1"])
    models["large"][race] = fit([r for r in rows if r["race"] != race])

frames = defaultdict(list)
for r in rows:
    frames[(r["batch"], r["frame"])].append(r)
pred = {key(r): {"Q": r["R0_35"]["vlm"]["number"]} for r in rows}
for variant in ("small", "large"):
    for items in frames.values():
        P = np.array([proba(models[variant][r["race"]], r) for r in items])
        S = P.copy()
        for i, r in enumerate(items):
            q = r["R0_35"]["vlm"]["number"]
            if q:
                S[i, q - 1] += W
            pred[key(r)][f"L.{variant}"] = (int(P[i].argmax()) + 1) if P[i].max() > 0 else 0
        ri, cj = linear_sum_assignment(-S)
        for i, j in zip(ri, cj):
            r = items[i]
            q = r["R0_35"]["vlm"]["number"]
            a = int(j) + 1
            agree = q != 0 and q == a
            pred[key(r)][f"C.{variant}"] = a
            pred[key(r)][f"A.{variant}"] = a if agree else 0
            pred[key(r)][f"B.{variant}"] = a if (agree and P[i, j] >= P_MIN) else 0

CONDS = ["Q"] + [f"{c}.{v}" for c in ("L", "C", "A", "B") for v in ("small", "large")]


def tally(sel, c):
    k = Counter("abstain" if pred[key(r)][c] == 0 else "correct" if pred[key(r)][c] == r["human_number"] else "wrong" for r in sel)
    ans = k["correct"] + k["wrong"]
    return {"n": len(sel), "correct": k["correct"], "wrong": k["wrong"], "abstain": k["abstain"], "precision": (k["correct"] / ans) if ans else None}


def paired(a, b, sel=rows):
    k = Counter()
    for r in sel:
        ca, cb = pred[key(r)][a] == r["human_number"], pred[key(r)][b] == r["human_number"]
        k["improved" if cb and not ca else "worsened" if ca and not cb else "same"] += 1
    n, m = k["improved"] + k["worsened"], min(k["improved"], k["worsened"])
    return {"improved": k["improved"], "worsened": k["worsened"], "sign_test_p": min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0}


primary = paired("C.small", "C.large")
hard = [r for r in rows if r["human_number"] in (7, 10, 11)]
summary = {
    "n": len(rows), "frames": len(frames),
    "training_size_per_fold": {v: {race: models[v][race][2] for race in races} for v in ("small", "large")},
    "overall": {c: tally(rows, c) for c in CONDS},
    "primary_C_small_vs_large": primary,
    "more_data_helps": bool(primary["improved"] > primary["worsened"] and primary["sign_test_p"] < 0.05),
    "paired": {"L.small -> L.large": paired("L.small", "L.large"), "A.small -> A.large": paired("A.small", "A.large"),
               "C small -> large, numbers 7/10/11 only": paired("C.small", "C.large", hard)},
    "by_number": {c: {n: tally([r for r in rows if r["human_number"] == n], c) for n in NUMS} for c in CONDS},
    "by_batch": {c: {b: tally([r for r in rows if r["batch"] == b], c) for b in ("1", "2A", "2B")} for c in CONDS},
    "confusion_C_large": {n: dict(Counter(pred[key(r)]["C.large"] for r in rows if r["human_number"] == n)) for n in NUMS},
}
OUT.mkdir(exist_ok=True)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
with open(OUT / "rows.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["batch", "id", "race", "frame", "human_number"] + CONDS)
    for r in rows:
        w.writerow([r["batch"], r["id"], r["race"], r["frame"], r["human_number"]] + [pred[key(r)][c] for c in CONDS])

print(f"riders {len(rows)}  frames {len(frames)}  train size per fold: small {min(summary['training_size_per_fold']['small'].values())}-{max(summary['training_size_per_fold']['small'].values())}, large {min(summary['training_size_per_fold']['large'].values())}-{max(summary['training_size_per_fold']['large'].values())}")
for c in CONDS:
    v = summary["overall"][c]
    print(f"  {c:8s} correct {v['correct']:3d} / wrong {v['wrong']:3d} / abstain {v['abstain']:3d}   precision {v['precision']:.3f}")
print("PRIMARY C.small -> C.large:", primary, "| more data helps:", summary["more_data_helps"])
for k, v in summary["paired"].items():
    print(f"  {k:40s} {v['improved']:3d} / {v['worsened']:3d}  p={v['sign_test_p']:.4f}")
print("correct by number:")
print("  n         " + " ".join(f"{summary['by_number']['Q'][n]['n']:>4d}" for n in NUMS))
for c in CONDS:
    print(f"  {c:8s} " + " ".join(f"{summary['by_number'][c][n]['correct']:>4d}" for n in NUMS))
print("C.large confusion:")
for n in NUMS:
    d = summary["confusion_C_large"][n]
    print(f"  true {n:2d}: " + ", ".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda x: -x[1])))
