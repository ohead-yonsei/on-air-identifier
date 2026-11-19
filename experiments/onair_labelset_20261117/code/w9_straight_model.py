"""far (2026-11-20): a leader model for the home straight, learned from official passing times on dense pictures.
usage: w9_straight_model.py explore                 leave one day out on the 15 days that have pictures of the last 37 s; prints the winner calls
       w9_straight_model.py fit OUT.joblib WINDOW   one model on all 15 days (training pictures: the last WINDOW seconds)
       w9_straight_model.py score MODEL.joblib OUT.json DAY [DAY ...]    winner calls with that model on other days

Why: in a third of the races the winner call has no candidate although the winner's cap is read in many pictures; the strip-trained
leader model seldom picks the front rider in the home straight (the strip is not on screen there, so it never learned that view).
Training rows: every sure rider (cap 0.7 or more, v11 classifier) in the pictures of the last WINDOW seconds of the races whose clock
agrees with the official time; answer = the rider's number is the leader at that moment by the official passing times
(fn_common.reference_full; at the finish, the winner). Same 19 features and model settings as the other position models.
The winner call is as in the winner experiment (w4_winner_guard.py), with this model's picks as the votes: most-named number of the
last 6 s, 5 pictures or more, kept when still in front in the last 2 s."""
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft
import w4_winner_guard as w4

D15 = ["20261115", "otherday", "otherday2", "otherday3", "otherday4", "otherday5", "otherday9", "otherday10", "otherday11",
       "otherday14", "otherday15", "otherday16", "otherday17", "otherday18", "otherday19"]
TRAINED = D15[:6]          # the days whose caps trained the classifier by hand labels
SURE = 0.7


def day_races(day):
    fc.FOLDER.setdefault(day, day)
    off = fc.official(day)
    for race, dr in sorted(w4.dense(day).items()):
        rank = {h["number"]: h["rank"] for h in off[race]["horses"] if h.get("rank")}
        yield race, dr, off[race], next((x for x, r in rank.items() if r == 1), None)


def rows(day, window):
    X, y = [], []
    for race, dr, off, win in day_races(day):
        if not dr["clock_agrees"]:
            continue
        for p in dr["pictures"]:
            if not (-0.05 <= p["to_finish"] <= window + 0.05) or not p["boxes"]:
                continue
            ref = fc.reference_full(off, dr["winner_time"] - max(p["to_finish"], 0.0))
            if ref is None:
                continue
            lead = min(win if p["to_finish"] <= 0.05 else ref["leader"], 11)
            who = ft.identify([p], [list(range(len(p["boxes"])))], memory=False)[0]
            F = ft.features(p["boxes"], [c for _, c in who])
            for r, (num, c) in enumerate(who):
                if num and c >= SURE:
                    X.append([float(v) for v in F[r]])
                    y.append(int(num == lead))
    return X, y


def fit(X, y):
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261120).fit(np.array(X), np.array(y))


def picks(pics, model, tau=0.5):
    out = []
    for p in pics:
        n = None
        if p["boxes"]:
            who = ft.identify([p], [list(range(len(p["boxes"])))], memory=False)[0]
            idx = [r for r, (num, c) in enumerate(who) if num and c >= SURE]
            if idx:
                F = np.array(ft.features(p["boxes"], [c for _, c in who]))[idx]
                pf = model.predict_proba(F)[:, 1]
                j = int(np.argmax(pf))
                if pf[j] >= tau:
                    n = who[idx[j]][0]
        out.append(n)
    return out


def call(pics, model, also=None, need=w4.NEED, tau=0.5):
    """(candidate, call). also = a second model whose picks count as votes too (a picture votes once per number)."""
    last = [p for p in pics if -0.05 <= p["to_finish"] <= w4.SPAN + 0.05]
    a = picks(last, model, tau)
    b = picks(last, also) if also is not None else [None] * len(last)
    c = Counter()
    for x, z in zip(a, b):
        for n in {x, z}:
            if n is not None:
                c[n] += 1
    top = c.most_common(1)
    if not top or top[0][1] < need:
        return None, None, dict(c)
    cand = top[0][0]
    seen, front = w4.in_front(pics, cand, w4.GUARD_T)
    return cand, (cand if seen >= w4.GUARD_SEEN and front >= w4.GUARD_SHARE * seen else None), dict(c)


def tally(results):
    k = Counter("none" if n is None else ("right" if n == win else "wrong") for n, win in results)
    return f"{k['right']}+{k['wrong']}w of {len(results)}"


def main():
    mode = sys.argv[1]
    if mode == "explore":
        data = {(d, w): rows(d, w) for d in D15 for w in (6.0, 12.0)}
        print("training rows per window:", {w: sum(len(data[(d, w)][1]) for d in D15) for w in (6.0, 12.0)},
              "share leader:", {w: round(float(np.mean([v for d in D15 for v in data[(d, w)][1]])), 3) for w in (6.0, 12.0)}, flush=True)
        res = {}
        for d in D15:
            for w in (6.0, 12.0):
                X = [x for o in D15 if o != d for x in data[(o, w)][0]]
                y = [v for o in D15 if o != d for v in data[(o, w)][1]]
                m = fit(X, y)
                for race, dr, off, win in day_races(d):
                    for name, kw in ((f"straight {w:g}s", {}), (f"straight {w:g}s or strip", {"also": w4.STRIP}), (f"straight {w:g}s need 7", {"need": 7}), (f"straight {w:g}s tau 0.6", {"tau": 0.6})):
                        res.setdefault(name, {}).setdefault(d, []).append((call(dr["pictures"], m, **kw)[1], win))
            for race, dr, off, win in day_races(d):
                res.setdefault("strip (current)", {}).setdefault(d, []).append((w4.call(dr["pictures"])[0], win))
            print("day done", d, flush=True)
        for name, by in res.items():
            allr = [x for d in D15 for x in by[d]]
            tr = [x for d in TRAINED for x in by[d]]
            other = [x for d in D15 if d not in TRAINED for x in by[d]]
            print(f"{name:28s} all {tally(allr)} | days that trained the classifier {tally(tr)} | the other nine {tally(other)} | " + " ".join(f"{d[-2:]}:{tally(by[d]).split(' of')[0]}" for d in D15))
    elif mode == "fit":
        out, window = sys.argv[2], float(sys.argv[3])
        X, y = [], []
        for d in D15:
            a, b = rows(d, window)
            X += a
            y += b
        joblib.dump(fit(X, y), out)
        Path(out).with_suffix(".json").write_text(json.dumps({"days": D15, "window": window, "rows": len(y), "share_leader": float(np.mean(y)), "sure": SURE}, indent=1))
        print("model on", len(y), "sure riders,", round(float(np.mean(y)), 3), "leaders")
    else:
        model, out, days = joblib.load(sys.argv[2]), sys.argv[3], sys.argv[4:]
        rs = []
        for d in days:
            for race, dr, off, win in day_races(d):
                cand, n, votes = call(dr["pictures"], model)
                cand2, n2, votes2 = call(dr["pictures"], model, also=w4.STRIP)
                old = w4.call(dr["pictures"])[0]
                rs.append({"day": d, "race": race, "clock_agrees": dr["clock_agrees"], "winner": win, "strip": old, "straight": n, "straight_or_strip": n2,
                           "candidate": cand, "votes": votes})
        for key, label in (("strip", "current rule (strip-trained model)"), ("straight", "home-straight model"), ("straight_or_strip", "either model votes")):
            for name, sel in (("all races run", rs), ("clock agrees", [r for r in rs if r["clock_agrees"]])):
                print(f"{label:36s} {name:14s} {tally([(r[key], r['winner']) for r in sel])}")
        for d in days:
            print(" ", d, " | ".join(f"{k} {tally([(r[k], r['winner']) for r in rs if r['day'] == d])}" for k in ("strip", "straight", "straight_or_strip")))
        Path(out).write_text(json.dumps({"days": days, "rows": rs}, indent=1))


if __name__ == "__main__":
    main()
