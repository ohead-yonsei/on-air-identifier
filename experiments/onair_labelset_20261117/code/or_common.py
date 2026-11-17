"""Shared pieces for the learned front-of-field rule (L1). See order2/DESIGN_KO.md."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
THR = 0.7           # sure = classifier probability of the assigned number >= 0.7
TAU, K = 0.6, 2     # L1: per picture, at most K sure riders whose learned probability of being in the leading three is >= TAU
FEATURES = ["rank_y", "rank_x_right", "rank_x_left", "rank_h", "rank_y/n", "rank_xr/n", "rank_h/n", "n", "yb", "cx", "h", "h/hmax", "yb-mean", "cx-mean",
            "cx_std", "yb_std", "corr_xy", "h_mean", "p", "qwen_agree"]


def riders(O, name, past):
    """One picture -> list of (number, classifier probability, rank from the bottom, features)."""
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
    ry, rxr, rxl, rh = (np.argsort(np.argsort(v)) for v in (-yb, -cx, cx, -h))
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


def load_day(d):
    O = ROOT / d
    frames = json.loads((O / "manifest.json").read_text())["frames"]
    return [(m, {tag: riders(O, m["frame"] + suf, past) for tag, suf, past in (("t", "", False), ("m05", "_m05", True), ("m10", "_m10", True))}) for m in frames]


def statements(day_data, score_fn, tau=TAU, k=K):
    """score_fn(feature matrix) -> probability of being in the leading three; None = hand rule (the two lowest boxes), i.e. T1.
    Returns {frame: set of numbers} after the 2-of-3 vote and the carry rule (numbers stated at both t-3 s and t-6 s)."""
    base = {}
    for m, pics in day_data:
        votes = Counter()
        for L in pics.values():
            sure = [r for r in L if r[1] >= THR]
            if score_fn is None:
                chosen = {r[0] for r in sure if r[2] < 2}
            elif sure:
                pr = score_fn(np.array([r[3] for r in sure]))
                chosen = {sure[i][0] for i in np.argsort(-pr)[:k] if pr[i] >= tau}
            else:
                chosen = set()
            for n in chosen:
                votes[n] += 1
        base[m["frame"]] = {n for n, v in votes.items() if v >= 2}
    out = {}
    for m, _ in day_data:
        race, t = m["frame"].split("_t")
        prev = [base.get(f"{race}_t{int(t) - b:03d}", set()) for b in (3, 6)]
        out[m["frame"]] = base[m["frame"]] | (prev[0] & prev[1])
    return out
