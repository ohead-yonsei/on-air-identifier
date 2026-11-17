"""Abstain-rule test on the reserved batch 2B. Fixed before any 2B crop was viewed. See batch2b/DESIGN_KO.md.

Train: batch 1 only, race-excluded. Test: batch2b/results/evidence.json.
Writes batch2b/results/summary.json and batch2b/results/rows.csv.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "batch2b"
W, P_MIN = 1.0, 0.1
NUMS = list(range(1, 12))


def usable(path):
    return [r for r in json.loads(path.read_text()) if r["human_number"] and not r["duplicate_of"]]


train_all = [r for r in usable(ROOT / "results" / "evidence.json") if r["R0_27"]["features"] is not None]
test = usable(B / "results" / "evidence.json")
assert all(r["human_number"] in NUMS for r in test)
races = sorted({r["race"] for r in test})


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


frames = defaultdict(list)
for r in test:
    frames[r["frame"]].append(r)
pred = {}
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
        a = int(j) + 1
        agree = q != 0 and q == a
        pred[r["id"]] = {"Q": q, "C": a, "A": a if agree else 0, "B": a if (agree and P[i, j] >= P_MIN) else 0,
                         "p_lr_assigned": float(P[i, j]),
                         "qwen_state": "agree" if agree else ("abstain" if q == 0 else "overridden")}

CONDS = ["Q", "C", "A", "B"]


def tally(sel, c):
    k = Counter("abstain" if pred[r["id"]][c] == 0 else "correct" if pred[r["id"]][c] == r["human_number"] else "wrong" for r in sel)
    ans = k["correct"] + k["wrong"]
    return {"n": len(sel), "correct": k["correct"], "wrong": k["wrong"], "abstain": k["abstain"],
            "precision": (k["correct"] / ans) if ans else None, "coverage": ans / len(sel) if sel else None}


answered = [r for r in test if pred[r["id"]]["Q"] != 0]
kept = [r for r in answered if pred[r["id"]]["A"] != 0]
dropped = [r for r in answered if pred[r["id"]]["A"] == 0]
qc = lambda sel: sum(pred[r["id"]]["Q"] == r["human_number"] for r in sel)
table = [[qc(kept), len(kept) - qc(kept)], [qc(dropped), len(dropped) - qc(dropped)]]
fisher_p = float(fisher_exact(table)[1]) if kept and dropped else 1.0
a = tally(test, "A")
crit1 = a["precision"] is not None and a["precision"] >= 0.85
crit2 = bool(dropped) and (qc(dropped) / len(dropped) < qc(kept) / len(kept)) and fisher_p < 0.05
summary = {
    "labels": "assistant drafts, not yet human-reviewed", "n_test": len(test), "n_frames": len(frames), "W": W, "P_MIN": P_MIN,
    "overall": {c: tally(test, c) for c in CONDS},
    "qwen_answers_kept_by_rule_A": {"n": len(kept), "qwen_correct": qc(kept)},
    "qwen_answers_dropped_by_rule_A": {"n": len(dropped), "qwen_correct": qc(dropped)},
    "fisher_exact_p": fisher_p,
    "criterion_1_precision_A_at_least_0.85": bool(crit1),
    "criterion_2_dropped_answers_worse_p_below_0.05": bool(crit2),
    "rule_A_worked": bool(crit1 and crit2),
    "by_qwen_state_C": {s: tally([r for r in test if pred[r["id"]]["qwen_state"] == s], "C") for s in ("agree", "abstain", "overridden")},
    "by_number": {c: {n: tally([r for r in test if r["human_number"] == n], c) for n in NUMS} for c in CONDS},
    "rule_A_by_assigned_number": {n: tally([r for r in test if pred[r["id"]]["A"] == n], "A") for n in NUMS},
}
(B / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
with open(B / "results" / "rows.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "race", "frame", "height_px", "human_number", "Q", "C", "A", "B", "p_lr_assigned", "qwen_state"])
    for r in test:
        p = pred[r["id"]]
        w.writerow([r["id"], r["race"], r["frame"], r["height_px"], r["human_number"], p["Q"], p["C"], p["A"], p["B"], round(p["p_lr_assigned"], 4), p["qwen_state"]])

print(f"test (batch 2B, draft labels) {len(test)}   frames {len(frames)}")
for c in CONDS:
    v = summary["overall"][c]
    pr = "n/a" if v["precision"] is None else f"{v['precision']:.3f}"
    print(f"  {c}  correct {v['correct']:3d} / wrong {v['wrong']:3d} / abstain {v['abstain']:3d}   precision {pr}   coverage {v['coverage']:.3f}")
print("Qwen answers kept by rule A:", summary["qwen_answers_kept_by_rule_A"], "| dropped:", summary["qwen_answers_dropped_by_rule_A"], "| Fisher p =", round(fisher_p, 5))
print("criterion 1 (precision A >= 0.85):", crit1, "| criterion 2 (dropped worse, p<0.05):", crit2, "| RULE A WORKED:", summary["rule_A_worked"])
print("combined method by Qwen state:")
for s, v in summary["by_qwen_state_C"].items():
    print(f"  {s:10s} correct {v['correct']:3d} / wrong {v['wrong']:3d}  (n={v['n']})")
print("answered correctly by number (n / Q / C / A / B):")
for n in NUMS:
    b = summary["by_number"]
    print(f"  {n:2d}: n={b['Q'][n]['n']:2d}  Q {b['Q'][n]['correct']:2d}  C {b['C'][n]['correct']:2d}  A {b['A'][n]['correct']:2d}  B {b['B'][n]['correct']:2d}")
print("rule A precision by the number it answered:")
print("  " + "  ".join(f"{n}:{v['correct']}/{v['correct'] + v['wrong']}" for n, v in summary["rule_A_by_assigned_number"].items()))
