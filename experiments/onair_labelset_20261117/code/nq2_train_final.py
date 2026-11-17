"""Train the two models of the fast configuration F1 (no Qwen) on the four days already seen. See night/DESIGN_KO.md.
Writes night/l1_fast.joblib (is this sure rider in the strip's first three?) and night/leader_fast.joblib (is it the strip's first?)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nq_common import FEATURES_FAST, load_day_fast  # noqa: E402
from or_common import ROOT, THR  # noqa: E402

DAYS = ["da/otherday2", "da/otherday3", "da/otherday4", "da/otherday5"]
X, y3, y1 = [], [], []
for d in DAYS:
    for m, pics in load_day_fast(d):
        for L in pics.values():
            for num, p, _, f in L:
                if p >= THR:
                    X.append(f)
                    y3.append(int(num in m["strip"][:3]))
                    y1.append(int(num == m["strip"][0]))
X = np.array(X)
info = {"days": DAYS, "sure_riders": len(y3), "features": FEATURES_FAST}
for name, y in (("l1_fast", y3), ("leader_fast", y1)):
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117).fit(X, np.array(y))
    joblib.dump(gb, ROOT / "night" / f"{name}.joblib")
    info[name] = {"share_positive": float(np.mean(y)), "importance": dict(zip(FEATURES_FAST, [round(float(v), 4) for v in gb.feature_importances_]))}
(ROOT / "night" / "fast_train.json").write_text(json.dumps(info, indent=1))
print("trained F1 models on", len(y3), "sure riders")
