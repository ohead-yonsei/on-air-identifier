"""caps3: the order model, the leader model and the finish-stretch leader model trained again on the numbers of another cap classifier.
usage: h7_train_models.py NAME        NAME = the classifier's name: da/DAY/cnn_probs_NAME (wk2_apply.py) and caps2/dense/NAME/last/DAY.json (h4_reprob.py)

Same days, answers, features and model settings as the models in use:
  order model, leader model       nq2_train_final.py: strip frames of 2026-11-07, 10-24, 10-17, 10-10; answer = in the strip's first three / the strip's first
  finish-stretch leader model     fn8_train_fs.py: the last 37 s of the six development days, answer from official passing times
Only the cap probabilities (and so which riders are sure, and the confidence feature) differ.
Writes caps2/models/{l1,leader,fs_leader}_NAME.joblib and caps2/models/train_NAME.json."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft

NAME = sys.argv[1]
STRIP_DAYS = ["da/otherday2", "da/otherday3", "da/otherday4", "da/otherday5"]
DEV = ["otherday", "otherday2", "otherday3", "otherday4", "otherday5", "20261115"]
THR = 0.7


def riders_fast(O, name, past):
    """as nq_common.riders_fast, with this classifier's probabilities"""
    rs = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
    if not rs:
        return []
    P = np.array(json.loads((O / f"cnn_probs_{NAME}" / f"{name}.json").read_text())["probs"])
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
        out.append((a[i] + 1, p, f))
    return out


def fit(X, y):
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117).fit(np.array(X), np.array(y))


out = ROOT / "caps2" / "models"
out.mkdir(parents=True, exist_ok=True)
X, y3, y1 = [], [], []
for d in STRIP_DAYS:
    O = ROOT / d
    for m in json.loads((O / "manifest.json").read_text())["frames"]:
        for suf, past in (("", False), ("_m05", True), ("_m10", True)):
            for num, p, f in riders_fast(O, m["frame"] + suf, past):
                if p >= THR:
                    X.append(f)
                    y3.append(int(num in m["strip"][:3]))
                    y1.append(int(num == m["strip"][0]))
joblib.dump(fit(X, y3), out / f"l1_{NAME}.joblib")
joblib.dump(fit(X, y1), out / f"leader_{NAME}.joblib")
info = {"name": NAME, "strip_days": STRIP_DAYS, "sure_riders": len(y3), "share_in_first_three": float(np.mean(y3)), "share_first": float(np.mean(y1))}
print("order and leader models:", len(y3), "sure riders", flush=True)

X1, t1 = [], []
for day in DEV:
    dense = json.loads((ROOT / "caps2" / "dense" / NAME / "last" / f"{day}.json").read_text())["races"]
    off = fc.official(day)
    for race, dr in sorted(dense.items()):
        if not dr["clock_agrees"]:
            continue
        pics = dr["pictures"]
        who = ft.identify(pics, ft.link(pics), memory=False)
        for p, w in zip(pics, who):
            if p["to_finish"] < -0.05 or not p["boxes"]:
                continue
            ref = fc.reference(off[race], dr["winner_time"] - p["to_finish"])
            if ref is None:
                continue
            F = ft.features(p["boxes"], [c for _, c in w])
            for r, (num, c) in enumerate(w):
                if not num or c < ft.SURE:
                    continue
                if (num == ref["leader"] and len(ref["leader_lenient"]) == 1) or num not in ref["leader_lenient"]:
                    X1.append(F[r])
                    t1.append(int(num == ref["leader"]))
joblib.dump(fit(X1, t1), out / f"fs_leader_{NAME}.joblib")
info.update(finish_days=DEV, finish_samples=len(t1), finish_positive=int(sum(t1)))
(out / f"train_{NAME}.json").write_text(json.dumps(info, indent=1))
print("finish-stretch leader model:", len(t1), "samples;", "saved models for", NAME, flush=True)
