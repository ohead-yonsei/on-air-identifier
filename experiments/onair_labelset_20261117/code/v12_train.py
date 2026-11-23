"""v12 stage 2: the two models of v12, learned on the development days with the official passing times as the answer.
usage: v12_train.py            (CPU; reads v12/dense/DAY.json of the development days, writes v12/models/)

Riders: numbered in each picture on its own (linear assignment of the cap probabilities, as the program does), kept when the cap
probability is 0.7 or more. Features: the 19 box features of the fast configuration (fn_tracks.features, the same list the program builds).
  top5  "is in the first five": 1 if the horse is in the computed first five at that moment, 0 if it is more than 3 m behind the fifth;
        riders in between are left out (the official order is 0.1 s, about 1.7 m, coarse).
  gap   metres behind the leader at that moment (capped at 40 m), a regression used only to put the named horses in order.
A cap read as 11 stands for every number from 11 up (they look the same); in a race with numbers above 11 it is given the best-placed
such horse, as in top5/RESULT_KO.md. Only races whose clock agrees with the official time are used; race time from 0 s to the finish.
Writes v12/models/{top5,gap}_all.joblib and one pair per development day left out ({top5,gap}_without_DAY.joblib), plus train_info.json."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft

DEV = ["otherday", "otherday2", "otherday3", "otherday4", "otherday5", "20261115"]
GAP_CAP, MARGIN = 40.0, fc.MARGIN_M


def truth(off_race, r):
    """Official state at race time r -> (rank_of(number), metres_behind_leader(number), metres behind the fifth) or None."""
    ref = fc.reference_full(off_race, r)
    if ref is None or len(ref["order"]) < 6:
        return None
    order, d = ref["order"], ref["to_go"]
    rank = {n: i + 1 for i, n in enumerate(order)}
    high = [n for n in order if n >= 11]

    def who(n):
        return (high[0] if high else None) if n >= 11 else (n if n in rank else None)
    return {"rank": lambda n: rank.get(who(n)) if who(n) else None,
            "behind": lambda n: None if who(n) is None else d[who(n)] - d[order[0]],
            "fifth": d[order[4]] - d[order[0]], "order": order, "to_go": d}


def pictures(day):
    """Yield (race, picture, who, F, truth) for the pictures of clock-agreeing races."""
    dense = json.loads((ROOT / "v12" / "dense" / f"{day}.json").read_text())["races"]
    off = fc.official(day)
    for race, dr in sorted(dense.items()):
        if not dr["clock_agrees"]:
            continue
        pics = dr["pictures"]
        whos = ft.identify(pics, [[0] * len(p["boxes"]) for p in pics], memory=False)
        for p, w in zip(pics, whos):
            r = dr["winner_time"] - p["to_finish"]
            if p["to_finish"] < -0.05 or r < 0 or not p["boxes"]:
                continue
            tr = truth(off[race], r)
            if tr is None:
                continue
            yield race, p, w, np.array(ft.features(p["boxes"], [c for _, c in w])), tr


def samples(day):
    X5, y5, XG, yG = [], [], [], []
    for race, p, w, F, tr in pictures(day):
        for i, (num, c) in enumerate(w):
            if not num or c < ft.SURE or tr["rank"](num) is None:
                continue
            b = tr["behind"](num)
            if tr["rank"](num) <= 5 or b > tr["fifth"] + MARGIN:
                X5.append(F[i])
                y5.append(int(tr["rank"](num) <= 5))
            XG.append(F[i])
            yG.append(min(b, GAP_CAP))
    return X5, y5, XG, yG


def fit5(X, y):
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117).fit(np.array(X), np.array(y))


def fitG(X, y):
    return GradientBoostingRegressor(loss="huber", n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117).fit(np.array(X), np.array(y))


def main():
    data = {d: samples(d) for d in DEV}
    out = ROOT / "v12" / "models"
    out.mkdir(parents=True, exist_ok=True)
    info = {"days": DEV, "per_day": {d: {"top5_samples": len(v[1]), "top5_positive": int(sum(v[1])), "gap_samples": len(v[3])} for d, v in data.items()}}
    for tag, days in [("all", DEV)] + [(f"without_{d}", [x for x in DEV if x != d]) for d in DEV]:
        X5 = [x for d in days for x in data[d][0]]
        y5 = [x for d in days for x in data[d][1]]
        XG = [x for d in days for x in data[d][2]]
        yG = [x for d in days for x in data[d][3]]
        joblib.dump(fit5(X5, y5), out / f"top5_{tag}.joblib")
        joblib.dump(fitG(XG, yG), out / f"gap_{tag}.joblib")
        print(tag, len(y5), "top5 samples", len(yG), "gap samples", flush=True)
    (out / "train_info.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
