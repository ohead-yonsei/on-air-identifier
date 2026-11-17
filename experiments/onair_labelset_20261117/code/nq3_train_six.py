"""Train the same four models on six days instead of four (adds 2026-10-05 and 2026-10-03, the two days used as tests so far).
Question: does more strip-labelled data help? See night/DESIGN_KO.md (addendum). Writes night/{l1,leader,l1_fast,leader_fast}_six.joblib."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nq_common import load_day_fast  # noqa: E402
from or_common import ROOT, THR, load_day  # noqa: E402

DAYS = ["da/otherday2", "da/otherday3", "da/otherday4", "da/otherday5", "da/otherday6", "da/otherday7"]
info = {"days": DAYS}
for tag, loader in (("", load_day), ("_fast", load_day_fast)):
    X, y3, y1 = [], [], []
    for d in DAYS:
        for m, pics in loader(d):
            for L in pics.values():
                for num, p, _, f in L:
                    if p >= THR:
                        X.append(f)
                        y3.append(int(num in m["strip"][:3]))
                        y1.append(int(num == m["strip"][0]))
    for name, y in ((f"l1{tag}_six", y3), (f"leader{tag}_six", y1)):
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117).fit(np.array(X), np.array(y))
        joblib.dump(gb, ROOT / "night" / f"{name}.joblib")
        info[name] = {"sure_riders": len(y), "share_positive": float(np.mean(y))}
(ROOT / "night" / "six_train.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info))
