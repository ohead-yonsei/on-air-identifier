"""Silks (racing colours) features for every rider box in the stripset frames, and a first look at
whether they tell horses apart. Development races only for the separability check.

Torso region: rows 35%-75% of the rider box, central 70% of its width. Same 21 colour features as the cap
classifier (background removed against the rider-box border). Writes stripset/silks/<name>.json.
usage: python x1_silks_features.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from s3_classify import clip, features  # noqa: E402

S = ROOT / "stripset"
(S / "silks").mkdir(exist_ok=True)
manifest = json.loads((S / "manifest.json").read_text())["frames"]
items = []
for m in manifest:
    items.append((m, m["frame"], S / "frames" / f"{m['frame']}.png", S / "out" / "D0" / f"{m['frame']}.json", m["pts"]))
    for tag, off in (("m05", 0.5), ("m10", 1.0), ("m15", 1.5)):
        n = f"{m['frame']}_{tag}"
        items.append((m, n, S / "frames_past" / f"{n}.png", S / "out_past" / "D0" / f"{n}.json", m["pts"] - off))


def torso(box, W, H):
    l, t, r, b = box
    w, h = r - l, b - t
    return clip([l + 0.15 * w, t + 0.35 * h, r - 0.15 * w, t + 0.75 * h], W, H)


for m, name, img, res, pts in items:
    out = S / "silks" / f"{name}.json"
    if out.exists():
        continue
    bgr = cv2.imread(str(img))
    H, W = bgr.shape[:2]
    riders = json.loads(res.read_text())["riders"]
    feats = []
    for r in riders:
        tb = torso(r["box"], W, H)
        feats.append(features(bgr, tb, r["box"]) if tb else None)
    out.write_text(json.dumps({"name": name, "frame": m["frame"], "race": m["race"], "split": m["split"], "pts": pts, "features": feats}))
print("silks files", len(list((S / "silks").glob("*.json"))))

# Separability on development races: within a race, are two boxes with the same cap-sure number closer in silks features
# than two boxes with different cap-sure numbers? (cap-sure numbers are about 88% right, so this is a noisy check)
rng = np.random.default_rng(0)
same, diff = [], []
by_race = {}
for m, name, img, res, pts in items:
    if m["split"] != "dev":
        continue
    riders = json.loads(res.read_text())["riders"]
    feats = json.loads((S / "silks" / f"{name}.json").read_text())["features"]
    for r, f in zip(riders, feats):
        if r["state"] == "sure" and f is not None:
            by_race.setdefault(m["race"], []).append((r["no"], np.array(f), name))
allf = np.array([f for v in by_race.values() for _, f, _ in v])
sd = allf.std(axis=0) + 1e-6
for race, v in by_race.items():
    idx = rng.permutation(len(v))[:400]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            na, fa, fra = v[idx[a]]
            nb, fb, frb = v[idx[b]]
            if fra == frb:
                continue  # different frames only
            d = float(np.linalg.norm((fa - fb) / sd))
            (same if na == nb else diff).append(d)
same, diff = np.array(same), np.array(diff)
ranks = np.argsort(np.argsort(np.concatenate([same, diff])))
auc = 1 - (ranks[: len(same)].sum() - len(same) * (len(same) - 1) / 2) / (len(same) * len(diff))
print(f"dev races: cap-sure instances {sum(len(v) for v in by_race.values())} | same-number pairs {len(same)}, different-number pairs {len(diff)}")
print(f"  distance, same number: median {np.median(same):.2f} (25-75%: {np.percentile(same, 25):.2f}-{np.percentile(same, 75):.2f})")
print(f"  distance, different  : median {np.median(diff):.2f} (25-75%: {np.percentile(diff, 25):.2f}-{np.percentile(diff, 75):.2f})")
print(f"  AUC (different farther than same): {auc:.3f}")
(S / "silks_scale.json").write_text(json.dumps({"sd": sd.tolist(), "note": "per-feature standard deviation over cap-sure instances of development races"}))
