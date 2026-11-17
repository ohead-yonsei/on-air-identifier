"""EXPLORATION on the four days the front-of-field model was trained on (not a test): learn "is this rider the leader?"
from the same 20 box features, label = the strip's first number. Leave-one-day-out. usage: ld1_explore.py"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ld_common import leader_statements, pick_hand, pick_model, tally  # noqa: E402
from or_common import ROOT, THR, load_day  # noqa: E402

DAYS = [("11-07", "da/otherday2"), ("10-24", "da/otherday3"), ("10-17", "da/otherday4"), ("10-10", "da/otherday5")]
data = {d: load_day(p) for d, p in DAYS}
l1 = joblib.load(ROOT / "order2" / "l1_model.joblib")


def show(name, rows):
    P = [r["right"] / max(1, r["statements"]) for r in rows]
    C = [r["right"] / r["points"] for r in rows]
    st, ok, n = sum(r["statements"] for r in rows), sum(r["right"] for r in rows), sum(r["points"] for r in rows)
    ok2 = sum(r["in_first_two"] for r in rows)
    print(f"{name:44s} | " + "  ".join(f"{p:.2f}/{c:.2f}" for p, c in zip(P, C)) + f" | pooled precision {ok / max(1, st):.3f} coverage {ok / n:.3f} (in first two {ok2 / max(1, st):.3f}, statements {st})")
    return ok / max(1, st), ok / n


print("rule | per day precision/coverage (" + ", ".join(d for d, _ in DAYS) + ")")
for carry in (False, True):
    show(f"hand rule (lowest box), carry={carry}", [tally(data[d], leader_statements(data[d], pick_hand, carry))[0] for d, _ in DAYS])
    for tau in (0.5, 0.6, 0.7, 0.8):
        show(f"front-of-field model argmax, tau {tau}, carry={carry}",
             [tally(data[d], leader_statements(data[d], lambda L, tau=tau: pick_model(L, lambda F: l1.predict_proba(F)[:, 1], tau), carry))[0] for d, _ in DAYS])
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
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117)
    gb.fit(np.array(X), np.array(y))
    models[day] = gb
    print(f"  trained without {day}: {len(y)} sure riders, {np.mean(y):.2f} are the strip's first")
res = {}
for carry in (False, True):
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
        res[(tau, carry)] = show(f"learned leader model, tau {tau}, carry={carry}",
                                 [tally(data[d], leader_statements(data[d], lambda L, d=d, tau=tau: pick_model(L, lambda F: models[d].predict_proba(F)[:, 1], tau), carry))[0] for d, _ in DAYS])
ok = {k: v for k, v in res.items() if v[0] >= 0.85}
print("configs with pooled precision >= 0.85:", sorted(ok.items(), key=lambda kv: -kv[1][1]))
print("best coverage overall:", max(res.items(), key=lambda kv: kv[1][1]))
