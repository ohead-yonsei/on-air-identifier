"""Shared pieces for the fast configuration without Qwen (F1). See night/DESIGN_KO.md.
Same riders and classifier probabilities as L1; the only differences: Qwen's colour answer is not used when numbers are assigned,
and the "Qwen agrees" feature is dropped (19 features instead of 20)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from or_common import FEATURES, ROOT  # noqa: E402

FEATURES_FAST = FEATURES[:-1]


def riders_fast(O, name, past):
    """One picture -> list of (number, classifier probability, rank from the bottom, 19 features). No Qwen anywhere."""
    rs = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
    if not rs:
        return []
    P = np.array(json.loads((O / "cnn_probs_two" / f"{name}.json").read_text())["probs"])
    ri, cj = linear_sum_assignment(-P)
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
             (yb[i] - yb.mean()) / 1080, (cx[i] - cx.mean()) / 1920, cx.std() / 1920, yb.std() / 1080, corr, h.mean() / 1080, p]
        out.append((a[i] + 1, p, int(ry[i]), f))
    return out


def load_day_fast(d):
    O = ROOT / d
    frames = json.loads((O / "manifest.json").read_text())["frames"]
    return [(m, {tag: riders_fast(O, m["frame"] + suf, past) for tag, suf, past in (("t", "", False), ("m05", "_m05", True), ("m10", "_m10", True))}) for m in frames]


def ab(day_data, said):
    """statements, in the strip's first three, time points, time points with at least one right statement"""
    st = ok = good = 0
    g = {}
    for m, _ in day_data:
        nums, B = said[m["frame"]], m["strip"]
        c = sum(n in B[:3] for n in nums)
        st += len(nums)
        ok += c
        g[m["frame"]] = c > 0
        good += c > 0
    return {"statements": st, "in_top3": ok, "points": len(day_data), "good_points": good}, g
