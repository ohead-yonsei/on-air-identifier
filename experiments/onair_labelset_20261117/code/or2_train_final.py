"""Train the final front-of-field model (L1) on the four days already seen. See order2/DESIGN_KO.md. Writes order2/l1_model.joblib."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from or_common import FEATURES, ROOT, THR, load_day  # noqa: E402

DAYS = ["da/otherday2", "da/otherday3", "da/otherday4", "da/otherday5"]
X, y = [], []
for d in DAYS:
    for m, pics in load_day(d):
        for L in pics.values():
            for num, p, _, f in L:
                if p >= THR:
                    X.append(f)
                    y.append(int(num in m["strip"][:3]))
gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117)
gb.fit(np.array(X), np.array(y))
joblib.dump(gb, ROOT / "order2" / "l1_model.joblib")
(ROOT / "order2" / "l1_train.json").write_text(json.dumps({"days": DAYS, "sure_riders": len(y), "share_in_leading_three": float(np.mean(y)), "features": FEATURES,
                                                             "importance": dict(zip(FEATURES, [round(float(v), 4) for v in gb.feature_importances_]))}, indent=1))
print("trained on", len(y), "sure riders from", DAYS)
