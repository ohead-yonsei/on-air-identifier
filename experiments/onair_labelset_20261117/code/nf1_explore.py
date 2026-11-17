"""EXPLORATION on the four training days (not a test): a fallback for time points where L1 says nothing.
Leave-one-day-out models, as in nq1_explore.py. usage: nf1_explore.py"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nf_common import statements_fallback  # noqa: E402
from nq_common import ab  # noqa: E402
from or_common import THR, load_day, statements  # noqa: E402

DAYS = [("11-07", "da/otherday2"), ("10-24", "da/otherday3"), ("10-17", "da/otherday4"), ("10-10", "da/otherday5")]
data = {d: load_day(p) for d, p in DAYS}
fns = {}
for leave, _ in DAYS:
    X, y = [], []
    for other, _ in DAYS:
        if other == leave:
            continue
        for m, pics in data[other]:
            for L in pics.values():
                for num, p, _, f in L:
                    if p >= THR:
                        X.append(f)
                        y.append(int(num in m["strip"][:3]))
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117).fit(np.array(X), np.array(y))
    fns[leave] = lambda F, gb=gb: gb.predict_proba(F)[:, 1]


def line(name, rows, add=None):
    st, ok, n, g = (sum(r[k] for r in rows) for k in ("statements", "in_top3", "points", "good_points"))
    extra = f" | added {add[0]}, right {add[1]} ({add[1] / max(1, add[0]):.2f})" if add else ""
    print(f"{name:30s} | " + "  ".join(f"A {r['in_top3'] / max(1, r['statements']):.2f} B {r['good_points'] / r['points']:.2f}" for r in rows) + f" | pooled A {ok / max(1, st):.3f} ({ok}/{st}) B {g / n:.3f} ({g}/{n})" + extra)


line("L1", [ab(data[d], statements(data[d], fns[d]))[0] for d, _ in DAYS])
for rule, t1 in (("hold", 0), ("single", 0.9), ("single", 0.8), ("single", 0.7), ("single", 0.6), ("single", 0.5), ("hold_then_single", 0.8), ("hold_then_single", 0.7), ("hold_then_single", 0.6)):
    rows, na, ra = [], 0, 0
    for d, _ in DAYS:
        said, added = statements_fallback(data[d], fns[d], rule, t1)
        rows.append(ab(data[d], said)[0])
        for m, _ in data[d]:
            for n in added[m["frame"]]:
                na += 1
                ra += n in m["strip"][:3]
    line(f"L1 + {rule} {t1 if 'single' in rule else ''}", rows, (na, ra))
