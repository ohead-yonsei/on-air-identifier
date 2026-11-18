"""Finish stretch 2, stage 2: order and leader models learned on the no-strip stretch, with the official passing times as the answer.
usage: fn8_train_fs.py

Data: the dense pictures (fn6_dense.py) of the development days, races whose clock agrees with the official time.
Riders: numbered in each picture on its own (as the fast configuration does), kept when the cap probability is 0.7 or more.
Answer for "is in the first three": 1 if the number is in the computed first three at that moment, 0 if it is more than 3 m behind the third;
riders in between are left out. Answer for "is first": 1 if first, 0 if more than 3 m behind the first; pictures with two horses within 3 m
of the front contribute only their clear negatives. Same model type and settings as the strip-trained models (nq2_train_final.py).
Writes finish2/models/fs_{order,leader}_all.joblib and one pair per development day left out (fs_*_without_DAY.joblib)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft

DEV = ["otherday", "otherday2", "otherday3", "otherday4", "otherday5", "20261115"]


def samples(day):
    dense = json.loads((ROOT / "finish2" / "dense" / f"{day}.json").read_text())["races"]
    off = fc.official(day)
    X3, y3, X1, y1 = [], [], [], []
    for race, dr in sorted(dense.items()):
        if not dr["clock_agrees"]:
            continue
        pics = dr["pictures"]
        who = ft.identify(pics, ft.link(pics), memory=False)
        for p, w in zip(pics, who):
            if p["to_finish"] < -0.05 or not p["boxes"]:
                continue
            ref = fc.reference(off[race], dr["winner_time"] - p["to_finish"])
            if ref is None:
                continue
            F = ft.features(p["boxes"], [c for _, c in w])
            for r, (num, c) in enumerate(w):
                if not num or c < ft.SURE:
                    continue
                if num in ref["top3"] or num not in ref["top3_lenient"]:
                    X3.append(F[r])
                    y3.append(int(num in ref["top3"]))
                if (num == ref["leader"] and len(ref["leader_lenient"]) == 1) or num not in ref["leader_lenient"]:
                    X1.append(F[r])
                    y1.append(int(num == ref["leader"]))
    return X3, y3, X1, y1


def fit(X, y):
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117).fit(np.array(X), np.array(y))


def main():
    data = {d: samples(d) for d in DEV}
    out = ROOT / "finish2" / "models"
    out.mkdir(parents=True, exist_ok=True)
    info = {"days": DEV, "per_day": {d: {"order_samples": len(v[1]), "order_positive": int(sum(v[1])), "leader_samples": len(v[3]), "leader_positive": int(sum(v[3]))} for d, v in data.items()}}
    for tag, days in [("all", DEV)] + [(f"without_{d}", [x for x in DEV if x != d]) for d in DEV]:
        X3 = [x for d in days for x in data[d][0]]
        y3 = [x for d in days for x in data[d][1]]
        X1 = [x for d in days for x in data[d][2]]
        y1 = [x for d in days for x in data[d][3]]
        joblib.dump(fit(X3, y3), out / f"fs_order_{tag}.joblib")
        joblib.dump(fit(X1, y1), out / f"fs_leader_{tag}.joblib")
        print(tag, "order", len(y3), "leader", len(y1), flush=True)
    (out / "fs_train.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
