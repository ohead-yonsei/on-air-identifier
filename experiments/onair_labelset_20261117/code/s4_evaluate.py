"""Stage 4: score every condition in DESIGN_KO.md against the human labels.

Reads results/evidence.json. Writes results/summary.json and results/rows.csv.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
NUMS = list(range(1, 12))
rows = [r for r in json.loads((ROOT / "results" / "evidence.json").read_text()) if r["human_number"] and not r["duplicate_of"]]
races = sorted({r["race"] for r in rows})
pred = {r["id"]: {} for r in rows}      # condition -> predicted number (0 = abstain)
score = {r["id"]: {} for r in rows}     # condition -> list of 11 scores (for assignment)

# ---- current rule and Qwen ----
for r in rows:
    for m, tag in (("R0_27", "H27"), ("R0_35", "H35")):
        s = r[m]["hsv"]["all_scores"]
        pred[r["id"]][tag + "_rule"] = r[m]["hsv"]["number"]            # with the rule's own abstain
        pred[r["id"]][tag + "_I"] = (int(np.argmax(s)) + 1) if max(s) > 0 else 0
        score[r["id"]][tag] = s
    pred[r["id"]]["Q35"] = r["R0_35"]["vlm"]["number"]

# ---- learned classifiers, leave one race out ----
for m, tag in (("R0_27", "L27"), ("R0_35", "L35")):
    for held in races:
        train = [r for r in rows if r["race"] != held and r[m]["features"] is not None]
        test = [r for r in rows if r["race"] == held]
        X = np.array([r[m]["features"] for r in train])
        y = np.array([r["human_number"] for r in train])
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, y)
        cents = {n: Xs[y == n].mean(axis=0) for n in sorted(set(y))}
        for r in test:
            f = r[m]["features"]
            if f is None:
                pred[r["id"]][tag + "-LR_I"] = pred[r["id"]][tag + "-NC"] = 0
                continue
            x = scaler.transform(np.array([f]))
            p = np.zeros(11)
            p[np.array(lr.classes_) - 1] = lr.predict_proba(x)[0]
            pred[r["id"]][tag + "-LR_I"] = int(np.argmax(p)) + 1
            score[r["id"]][tag + "-LR"] = p.tolist()
            pred[r["id"]][tag + "-NC"] = int(min(cents, key=lambda n: np.linalg.norm(x[0] - cents[n])))

# ---- within-frame one-to-one assignment over every scored rider of the frame ----
frames = defaultdict(list)
for r in rows:
    frames[r["frame"]].append(r)
for tag in ("H27", "H35", "L27-LR", "L35-LR"):
    for f, items in frames.items():
        have = [r for r in items if tag in score[r["id"]]]
        for r in items:
            pred[r["id"]][tag + "_A"] = 0
        if not have:
            continue
        M = np.array([score[r["id"]][tag] for r in have])
        ri, cj = linear_sum_assignment(-M)
        for i, j in zip(ri, cj):
            pred[have[i]["id"]][tag + "_A"] = (j + 1) if M[i, j] > 0 else 0

CONDS = ["H27_rule", "H35_rule", "Q35", "H27_I", "H27_A", "H35_I", "H35_A",
         "L27-NC", "L35-NC", "L27-LR_I", "L27-LR_A", "L35-LR_I", "L35-LR_A"]


def verdict(p, t):
    return "abstain" if p == 0 else ("correct" if p == t else "wrong")


def tally(sel, c):
    k = Counter(verdict(pred[r["id"]][c], r["human_number"]) for r in sel)
    return {"n": len(sel), "correct": k["correct"], "wrong": k["wrong"], "abstain": k["abstain"]}


def paired(sel, a, b):
    k = Counter()
    for r in sel:
        ca, cb = pred[r["id"]][a] == r["human_number"], pred[r["id"]][b] == r["human_number"]
        k["improved" if cb and not ca else "worsened" if ca and not cb else "same"] += 1
    return {"improved": k["improved"], "worsened": k["worsened"], "same": k["same"]}


big = [r for r in rows if len(frames[r["frame"]]) >= 5]
summary = {
    "n_riders": len(rows), "n_frames": len(frames), "n_riders_in_frames_of_5_plus": len(big),
    "overall": {c: tally(rows, c) for c in CONDS},
    "frames_of_5_plus": {c: tally(big, c) for c in CONDS},
    "by_number": {c: {n: tally([r for r in rows if r["human_number"] == n], c) for n in NUMS}
                  for c in ("H35_rule", "Q35", "H35_I", "L27-LR_I", "L35-LR_I", "L27-LR_A", "L35-LR_A")},
    "by_race": {c: {race: tally([r for r in rows if r["race"] == race], c) for race in races}
                for c in ("H35_rule", "Q35", "L27-LR_I", "L35-LR_I", "L35-LR_A")},
    "paired": {f"{a} -> {b}": paired(rows, a, b) for a, b in [
        ("H27_I", "H27_A"), ("H35_I", "H35_A"), ("L27-LR_I", "L27-LR_A"), ("L35-LR_I", "L35-LR_A"),
        ("H35_I", "L35-LR_I"), ("H27_I", "L27-LR_I"), ("Q35", "L35-LR_I"), ("Q35", "L35-LR_A"), ("Q35", "L27-LR_A")]},
    "confusion": {c: {n: dict(Counter(pred[r["id"]][c] for r in rows if r["human_number"] == n)) for n in NUMS}
                  for c in ("Q35", "H35_I", "L35-LR_I", "L27-LR_I")},
}
(ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
with open(ROOT / "results" / "rows.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "race", "frame", "height_px", "human_number"] + CONDS)
    for r in rows:
        w.writerow([r["id"], r["race"], r["frame"], r["height_px"], r["human_number"]] + [pred[r["id"]][c] for c in CONDS])

f3 = lambda d: f"{d['correct']:3d}/{d['wrong']:3d}/{d['abstain']:3d}"
print(f"riders {len(rows)}  frames {len(frames)}  riders in frames of 5+ {len(big)}")
print("\ncondition      all (correct/wrong/abstain)   frames of 5+")
for c in CONDS:
    print(f"  {c:11s}  {f3(summary['overall'][c])}                 {f3(summary['frames_of_5_plus'][c])}")
print("\npaired (improved / worsened):")
for k, v in summary["paired"].items():
    print(f"  {k:24s} {v['improved']:3d} / {v['worsened']:3d}")
print("\ncorrect by number (n):")
print("  cond        " + " ".join(f"{n:>5d}" for n in NUMS))
print("  n           " + " ".join(f"{summary['by_number']['Q35'][n]['n']:>5d}" for n in NUMS))
for c in summary["by_number"]:
    print(f"  {c:11s} " + " ".join(f"{summary['by_number'][c][n]['correct']:>5d}" for n in NUMS))
print("\ncorrect by race:")
print("  cond        " + " ".join(f"{r:>5s}" for r in races))
print("  n           " + " ".join(f"{summary['by_race']['Q35'][r]['n']:>5d}" for r in races))
for c in summary["by_race"]:
    print(f"  {c:11s} " + " ".join(f"{summary['by_race'][c][r]['correct']:>5d}" for r in races))
