"""EXPLORATION on days already seen (not a test): learn "is this rider in the leading three?" from box geometry,
instead of the hand rule "the two lowest boxes". Training signal: the strip (number in the strip's first three).
Leave-one-day-out over four days with adaptive detection. usage: or1_learn_order.py"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
DAYS = [("11-07", "da/otherday2"), ("10-24", "da/otherday3"), ("10-17", "da/otherday4"), ("10-10", "da/otherday5")]
THR = 0.7


def riders(O, name, past):
    rs = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
    if not rs:
        return []
    P = np.array(json.loads((O / "cnn_probs_two" / f"{name}.json").read_text())["probs"])
    Sc = P.copy()
    q = [r["qwen_number"] for r in rs]
    for i, n in enumerate(q):
        if n:
            Sc[i, n - 1] += 1.0
    ri, cj = linear_sum_assignment(-Sc)
    a = {int(i): int(j) for i, j in zip(ri, cj)}
    boxes = np.array([r["box"] for r in rs], float)
    cx, yb, h = (boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3], boxes[:, 3] - boxes[:, 1]
    n = len(rs)
    ry = np.argsort(np.argsort(-yb))            # 0 = lowest box
    rxr = np.argsort(np.argsort(-cx))           # 0 = rightmost
    rxl = np.argsort(np.argsort(cx))            # 0 = leftmost
    rh = np.argsort(np.argsort(-h))             # 0 = tallest
    corr = float(np.corrcoef(cx, yb)[0, 1]) if n >= 3 and cx.std() > 1 and yb.std() > 1 else 0.0
    out = []
    for i in range(n):
        if i not in a:
            continue
        p = float(P[i, a[i]])
        f = [ry[i], rxr[i], rxl[i], rh[i], ry[i] / n, rxr[i] / n, rh[i] / n, n, yb[i] / 1080, cx[i] / 1920, h[i] / 1080, h[i] / h.max(),
             (yb[i] - yb.mean()) / 1080, (cx[i] - cx.mean()) / 1920, cx.std() / 1920, yb.std() / 1080, corr, h.mean() / 1080, p, float(q[i] == a[i] + 1)]
        out.append((a[i] + 1, p, int(ry[i]), f))
    return out


data = {}
for day, d in DAYS:
    O = ROOT / d
    frames = json.loads((O / "manifest.json").read_text())["frames"]
    data[day] = [(m, {tag: riders(O, m["frame"] + suf, past) for tag, suf, past in (("t", "", False), ("m05", "_m05", True), ("m10", "_m10", True))}) for m in frames]


def evaluate(day, score_fn, tau, K):
    """score_fn(features) -> probability of being in the leading three; None = hand rule (two lowest boxes)."""
    frames = data[day]
    said_at = {}
    for m, pics in frames:
        votes = Counter()
        for L in pics.values():
            sure = [r for r in L if r[1] >= THR]
            if score_fn is None:
                chosen = {r[0] for r in L if r[2] < 2 and r[1] >= THR}
            else:
                if not sure:
                    continue
                pr = score_fn(np.array([r[3] for r in sure]))
                order = np.argsort(-pr)[:K]
                chosen = {sure[i][0] for i in order if pr[i] >= tau}
            for n in chosen:
                votes[n] += 1
        said_at[m["frame"]] = {n for n, v in votes.items() if v >= 2}
    st = ok = good = 0
    for m, _ in frames:
        race, t = m["frame"].split("_t")
        prev = [said_at.get(f"{race}_t{int(t) - b:03d}", set()) for b in (3, 6)]
        said = said_at[m["frame"]] | (prev[0] & prev[1])
        st += len(said)
        ok += sum(n in m["strip"][:3] for n in said)
        good += any(n in m["strip"][:3] for n in said)
    return ok, st, good, len(frames)


def show(name, rows):
    A = [r[0] / max(1, r[1]) for r in rows]
    B = [r[2] / r[3] for r in rows]
    print(f"{name:34s} | " + "  ".join(f"{a:.2f}/{b:.2f}" for a, b in zip(A, B)) + f" | pooled {sum(r[0] for r in rows) / max(1, sum(r[1] for r in rows)):.3f}/{sum(r[2] for r in rows) / sum(r[3] for r in rows):.3f}")


print("rule | per day A/B (" + ", ".join(d for d, _ in DAYS) + ")")
show("hand rule (T1)", [evaluate(day, None, 0, 2) for day, _ in DAYS])
models = {}
for day, _ in DAYS:
    X, y = [], []
    for other, _ in DAYS:
        if other == day:
            continue
        for m, pics in data[other]:
            for L in pics.values():
                for num, p, _, f in L:
                    if p >= THR:
                        X.append(f)
                        y.append(int(num in m["strip"][:3]))
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117)
    gb.fit(np.array(X), np.array(y))
    models[day] = gb
    print(f"  trained without {day}: {len(y)} sure riders, {np.mean(y):.2f} in the leading three")
for K in (2, 3, 4):
    for tau in (0.4, 0.5, 0.6, 0.7, 0.8):
        show(f"learned, K{K}, tau {tau}", [evaluate(day, lambda F, day=day: models[day].predict_proba(F)[:, 1], tau, K) for day, _ in DAYS])
gb = models[DAYS[0][0]]
names = ["rank_y", "rank_x_right", "rank_x_left", "rank_h", "rank_y/n", "rank_xr/n", "rank_h/n", "n", "yb", "cx", "h", "h/hmax", "yb-mean", "cx-mean", "cx_std", "yb_std", "corr_xy", "h_mean", "p", "qwen_agree"]
print("feature importance (one model):", sorted(zip(np.round(gb.feature_importances_, 3), names), reverse=True)[:8])
