"""Race-excluded colour classifiers for honest testing: one model per race, trained without that race.

Same data and settings as train_final.py. Writes identifier/models_loro/excl_<race>.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
rows = []
for path in (ROOT / "results" / "evidence.json", ROOT / "batch2" / "results" / "evidence.json", ROOT / "batch2b" / "results" / "evidence.json"):
    rows += [r for r in json.loads(path.read_text()) if r["human_number"] and not r["duplicate_of"] and r["R0_27"]["features"] is not None]
out = ROOT / "identifier" / "models_loro"
out.mkdir(exist_ok=True)
for race in sorted({r["race"] for r in rows}):
    sel = [r for r in rows if r["race"] != race]
    X = np.array([r["R0_27"]["features"] for r in sel])
    y = np.array([r["human_number"] for r in sel])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
    (out / f"excl_{race}.json").write_text(json.dumps({
        "what": f"as model.json but trained without race {race}", "trained_on": {"riders": len(sel)},
        "classes": [int(c) for c in lr.classes_], "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()}))
    print(race, "trained on", len(sel))
