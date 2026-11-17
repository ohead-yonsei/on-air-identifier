"""EXPLORATION on the four training days (not a test): for two numbers the identifier states together, which one is ahead?
Scores compared: how low the box is (hand), the front-of-field model, a leave-one-day-out leader model. usage: ld1b_explore_pairs.py"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ld_common import pair_statements, pair_tally, yb_score  # noqa: E402
from or_common import ROOT, THR, load_day, statements  # noqa: E402

DAYS = [("11-07", "da/otherday2"), ("10-24", "da/otherday3"), ("10-17", "da/otherday4"), ("10-10", "da/otherday5")]
data = {d: load_day(p) for d, p in DAYS}
l1 = joblib.load(ROOT / "order2" / "l1_model.joblib")
l1fn = lambda F: l1.predict_proba(F)[:, 1]  # noqa: E731
stated = {d: statements(data[d], l1fn) for d, _ in DAYS}
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
                        y.append(int(num == m["strip"][0]))
    models[day] = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117).fit(np.array(X), np.array(y))


def show(name, fn_for_day, margin):
    rows, cand = [], 0
    for d, _ in DAYS:
        pairs, c = pair_statements(data[d], stated[d], fn_for_day(d), margin)
        cand += c
        rows.append(pair_tally(data[d], pairs)[0])
    sc, ok, said = (sum(r[k] for r in rows) for k in ("scoreable", "right", "said"))
    print(f"{name:34s} margin {margin:<5} | " + "  ".join(f"{r['right']}/{r['scoreable']}" for r in rows) + f" | pooled right {ok / max(1, sc):.3f} of scoreable {sc}; said {said} of {cand} candidate pairs ({said / max(1, cand):.2f})")


print("score | per day right/scoreable (" + ", ".join(d for d, _ in DAYS) + ")")
for mg in (0.0, 0.02, 0.04, 0.06, 0.1):
    show("hand: lower box", lambda d: yb_score, mg)
for mg in (0.0, 0.05, 0.1, 0.2, 0.3):
    show("front-of-field model", lambda d: l1fn, mg)
for mg in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
    show("leader model (leave one day out)", lambda d: (lambda F: models[d].predict_proba(F)[:, 1]), mg)
