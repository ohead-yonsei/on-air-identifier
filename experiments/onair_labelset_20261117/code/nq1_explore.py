"""EXPLORATION on the four training days (not a test): does the identifier still work without Qwen?
Both configurations are retrained leave-one-day-out so the comparison is like for like. usage: nq1_explore.py"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import or_common  # noqa: E402
from ld_common import leader_statements, pair_statements, pair_tally, pick_model, tally  # noqa: E402
from nq_common import ab, load_day_fast  # noqa: E402
from or_common import load_day, statements  # noqa: E402

DAYS = [("11-07", "da/otherday2"), ("10-24", "da/otherday3"), ("10-17", "da/otherday4"), ("10-10", "da/otherday5")]
data = {"L1": {d: load_day(p) for d, p in DAYS}, "F1": {d: load_day_fast(p) for d, p in DAYS}}


def fit(cfg, leave, thr, label):
    X, y = [], []
    for other, _ in DAYS:
        if other == leave:
            continue
        for m, pics in data[cfg][other]:
            for L in pics.values():
                for num, p, _, f in L:
                    if p >= thr:
                        X.append(f)
                        y.append(int(label(num, m["strip"])))
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117).fit(np.array(X), np.array(y))


def line(name, rows):
    st, ok, n, g = (sum(r[k] for r in rows) for k in ("statements", "in_top3", "points", "good_points"))
    print(f"{name:28s} | " + "  ".join(f"A {r['in_top3'] / max(1, r['statements']):.2f} B {r['good_points'] / r['points']:.2f}" for r in rows) + f" | pooled A {ok / max(1, st):.3f} ({ok}/{st}) B {g / n:.3f} ({g}/{n})")


print("configuration | per day (" + ", ".join(d for d, _ in DAYS) + ")")
for cfg, thr in (("L1", 0.7), ("F1", 0.7), ("F1", 0.8), ("F1", 0.6)):
    or_common.THR = thr          # the sure threshold used inside statements()
    import ld_common
    ld_common.THR = thr
    rows, lrows, prows, cand = [], [], [], 0
    for d, _ in DAYS:
        top = fit(cfg, d, thr, lambda num, strip: num in strip[:3])
        first = fit(cfg, d, thr, lambda num, strip: num == strip[0])
        fn = lambda F, m=top: m.predict_proba(F)[:, 1]  # noqa: E731
        ffn = lambda F, m=first: m.predict_proba(F)[:, 1]  # noqa: E731
        said = statements(data[cfg][d], fn)
        rows.append(ab(data[cfg][d], said)[0])
        lrows.append(tally(data[cfg][d], leader_statements(data[cfg][d], lambda P, ffn=ffn: pick_model(P, ffn, 0.6), True))[0])
        pairs, c = pair_statements(data[cfg][d], said, ffn, 0.2)
        cand += c
        prows.append(pair_tally(data[cfg][d], pairs)[0])
    line(f"{cfg} sure>={thr}", rows)
    ls, lo, ln = (sum(r[k] for r in lrows) for k in ("statements", "right", "points"))
    ps, po = (sum(r[k] for r in prows) for k in ("scoreable", "right"))
    print(f"    leader {lo}/{ls} = {lo / max(1, ls):.3f}, said at {ls / ln:.2f} of points;  pairs {po}/{ps} = {po / max(1, ps):.3f}, said {sum(r['said'] for r in prows)} of {cand}")
or_common.THR = 0.7
