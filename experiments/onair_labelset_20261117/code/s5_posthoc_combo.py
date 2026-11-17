"""POST-HOC exploration (designed after seeing the stage-4 results; not part of DESIGN_KO.md).

Idea: Qwen is right about 80% of the time on seven colours but never names black, brown or
sky blue. The learned classifier is weaker overall but does separate those. Combine them:

    score[number] = p_LR27[number] + W * 1[Qwen named this number]      (W fixed at 1.0)

then either take the best number per rider (C_I) or assign distinct numbers within a frame (C_A).
The LR probabilities are leave-one-race-out, exactly as in stage 4. W is not tuned.
Because the idea came from looking at these same 147 riders, treat the numbers as a lead to
re-test on new data, not as a result.
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
rows = [r for r in json.loads((ROOT / "results" / "evidence.json").read_text()) if r["human_number"] and not r["duplicate_of"]]
races = sorted({r["race"] for r in rows})
P = {}
for held in races:
    train = [r for r in rows if r["race"] != held and r["R0_27"]["features"] is not None]
    X = np.array([r["R0_27"]["features"] for r in train])
    y = np.array([r["human_number"] for r in train])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
    for r in rows:
        if r["race"] != held:
            continue
        p = np.zeros(11)
        if r["R0_27"]["features"] is not None:
            p[np.array(lr.classes_) - 1] = lr.predict_proba(sc.transform(np.array([r["R0_27"]["features"]])))[0]
        P[r["id"]] = p

S, pred = {}, {r["id"]: {} for r in rows}
for r in rows:
    s = P[r["id"]].copy()
    q = r["R0_35"]["vlm"]["number"]
    if q:
        s[q - 1] += W
    S[r["id"]] = s
    pred[r["id"]]["Q35"] = q
    pred[r["id"]]["L27-LR_I"] = int(np.argmax(P[r["id"]])) + 1
    pred[r["id"]]["C_I"] = int(np.argmax(s)) + 1
frames = defaultdict(list)
for r in rows:
    frames[r["frame"]].append(r)
for f, items in frames.items():
    M = np.array([S[r["id"]] for r in items])
    ri, cj = linear_sum_assignment(-M)
    for i, j in zip(ri, cj):
        pred[items[i]["id"]]["C_A"] = int(j) + 1


def tally(sel, c):
    k = Counter("abstain" if pred[r["id"]][c] == 0 else "correct" if pred[r["id"]][c] == r["human_number"] else "wrong" for r in sel)
    return {"n": len(sel), "correct": k["correct"], "wrong": k["wrong"], "abstain": k["abstain"]}


def paired(a, b):
    k = Counter()
    for r in rows:
        ca, cb = pred[r["id"]][a] == r["human_number"], pred[r["id"]][b] == r["human_number"]
        k["improved" if cb and not ca else "worsened" if ca and not cb else "same"] += 1
    return dict(k)


CONDS = ["Q35", "L27-LR_I", "C_I", "C_A"]
out = {"status": "post-hoc exploration", "W": W,
       "overall": {c: tally(rows, c) for c in CONDS},
       "by_number": {c: {n: tally([r for r in rows if r["human_number"] == n], c) for n in range(1, 12)} for c in CONDS},
       "by_race": {c: {race: tally([r for r in rows if r["race"] == race], c) for race in races} for c in CONDS},
       "paired": {"Q35 -> C_A": paired("Q35", "C_A"), "C_I -> C_A": paired("C_I", "C_A"), "L27-LR_I -> C_A": paired("L27-LR_I", "C_A")},
       "confusion_C_A": {n: dict(Counter(pred[r["id"]]["C_A"] for r in rows if r["human_number"] == n)) for n in range(1, 12)},
       "predictions": {r["id"]: {"human": r["human_number"], **pred[r["id"]]} for r in rows}}
(ROOT / "results" / "posthoc_combo.json").write_text(json.dumps(out, indent=1))
for c in CONDS:
    v = out["overall"][c]
    print(f"{c:9s} correct {v['correct']:3d} / wrong {v['wrong']:3d} / abstain {v['abstain']:3d}")
print("paired:", out["paired"])
print("correct by number:")
print("  n        " + " ".join(f"{out['by_number']['Q35'][n]['n']:>4d}" for n in range(1, 12)))
for c in CONDS:
    print(f"  {c:8s} " + " ".join(f"{out['by_number'][c][n]['correct']:>4d}" for n in range(1, 12)))
print("correct by race:")
print("  n        " + " ".join(f"{out['by_race']['Q35'][r]['n']:>4d}" for r in races))
for c in CONDS:
    print(f"  {c:8s} " + " ".join(f"{out['by_race'][c][r]['correct']:>4d}" for r in races))
print("C_A confusion:")
for n in range(1, 12):
    d = out["confusion_C_A"][n]
    print(f"  true {n:2d}: " + ", ".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda x: -x[1])))
