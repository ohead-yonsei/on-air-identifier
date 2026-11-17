"""Fit the final colour classifier on every labelled rider and save it as plain JSON (no pickle).

Uses the 21 colour features of the R0_27 cap region stored in the three evidence files.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
rows = []
for batch, path in (("1", ROOT / "results" / "evidence.json"), ("2A", ROOT / "batch2" / "results" / "evidence.json"),
                    ("2B", ROOT / "batch2b" / "results" / "evidence.json")):
    for r in json.loads(path.read_text()):
        if r["human_number"] and not r["duplicate_of"] and r["R0_27"]["features"] is not None:
            rows.append((batch, r))
X = np.array([r["R0_27"]["features"] for _, r in rows])
y = np.array([r["human_number"] for _, r in rows])
sc = StandardScaler().fit(X)
lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)


def proba(x):
    z = ((x - sc.mean_) / sc.scale_) @ lr.coef_.T + lr.intercept_
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


assert np.abs(proba(X) - lr.predict_proba(sc.transform(X))).max() < 1e-9  # the JSON form reproduces sklearn exactly
model = {
    "what": "multinomial logistic regression on 21 colour features of the R0_27 cap region",
    "trained_on": {"riders": len(rows), "by_batch": dict(Counter(b for b, _ in rows)), "by_number": {int(k): int(v) for k, v in sorted(Counter(y.tolist()).items())},
                   "source": "2026-09-06 Seoul R1-R10 broadcast, human-confirmed cap numbers"},
    "classes": [int(c) for c in lr.classes_], "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
    "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
}
(ROOT / "identifier" / "model.json").write_text(json.dumps(model))
print("trained on", len(rows), "riders;", dict(Counter(b for b, _ in rows)), "| classes", model["classes"])
