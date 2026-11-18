"""finish3 stage 2 (exploration on days seen before): a leader model learned on the WHOLE race with official passing times as the answer.
usage: g5_explore.py OUT.json            (CPU only; reads finish3/dense/DAY.json from g4_dense_whole.py)

Why: the leader of v9/v10 comes from a model taught by the strip (so only the middle of the race), swapped for another model in the last 36 s.
It names no leader at about a third of the time points and is weak where the camera view is unlike the taught ones.
Candidates (same model type and settings as the earlier ones; riders numbered in each picture on its own, cap probability 0.7 or more):
  W19   one model for the whole race, the 19 box features
  W21   W19 plus two features that say where in the race the picture is: seconds since the clock started, and that divided by the usual
        time for the distance (both known to a live system: the clock is on screen, the distance is on the race card)
Answer while training: 1 if the rider's number is first at that moment, 0 if more than 3 m behind the first (as fn8_train_fs.py).
Statement rule (unchanged): a picture names the sure rider with the highest score if the score reaches the threshold; "leader" at a time
point = named in 2 of the 3 pictures (t, t-0.5, t-1.0 s), or carried when the two time points before agree. Time points: the finish and
every 3 s before it back to race time 9 s (as fn10_whole.py).
Base = v10: the strip-trained model before the last 36 s, the finish-stretch model (fn8) in the last 36 s, threshold 0.5.

Two sets of days. DEV (six development days): every model that is trained here leaves the scored day out; the finish-stretch model too.
The strip-trained model saw four of these days, so the base is flattered on DEV. HELD (the six measurement and confirmation days of
finish2): models trained on all DEV days; no model here was trained on them.
Rule for choosing (written before the first run): among W19/W21 x thresholds 0.3..0.7, on DEV, the highest "leader right share" over the
whole race among those whose "leader wrong share" is not above the base's on DEV. The choice is then read on HELD; a candidate goes to
the new days only if on HELD its right share is above the base's (paired interval above 0) and its wrong share is at most 2 points higher.
Main analysis: races whose clock agrees with the official time."""
import json
import random
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

DEV = ["otherday", "otherday2", "otherday3", "otherday4", "otherday5", "20261115"]
HELD = ["otherday6", "otherday7", "otherday8", "otherday9", "otherday10", "otherday11"]
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
PARTS = ("start", "middle", "finish", "whole race")


def expected_seconds(d):
    return d * (0.062 if d <= 1200 else 0.0635 if d <= 1400 else 0.066)


def load(day):
    """Pictures of a day: {race: {tau: picture}}, picture = {"nums": [...], "X": [[21 features]...], "ref": reference or None}."""
    dense = json.loads((ROOT / "finish3" / "dense" / f"{day}.json").read_text())["races"]
    off, out = fc.official(day), {}
    for race, dr in sorted(dense.items()):
        exp, pics = expected_seconds(dr["distance"]), {}
        for p in dr["pictures"]:
            e = {"nums": [], "X": [], "ref": fc.reference_full(off[race], dr["winner_time"] - p["to_finish"])}
            if p["boxes"]:
                who = ft.identify([p], [list(range(len(p["boxes"])))], memory=False)[0]
                F = ft.features(p["boxes"], [c for _, c in who])
                elapsed = p["t"] - (dr["clock_S"] - 0.1)
                for r, (num, c) in enumerate(who):
                    if num and c >= ft.SURE:
                        e["nums"].append(num)
                        e["X"].append([float(v) for v in F[r]] + [elapsed, elapsed / exp])
            pics[p["to_finish"]] = e
        out[race] = {"agree": dr["clock_agrees"], "tw": dr["winner_time"], "pics": pics}
    return out


def samples(data, days):
    X, y = [], []
    for d in days:
        for rr in data[d].values():
            if not rr["agree"]:
                continue
            for e in rr["pics"].values():
                ref = e["ref"]
                if ref is None:
                    continue
                for num, x in zip(e["nums"], e["X"]):
                    if (num == ref["leader"] and len(ref["leader_lenient"]) == 1) or num not in ref["leader_lenient"]:
                        X.append(x)
                        y.append(int(num == ref["leader"]))
    return np.array(X), np.array(y)


def fit(X, y):
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20261117).fit(X, y)


def score_day(data_day, models):
    """Adds e["s"][name] = scores of the sure riders of every picture, one predict call per model."""
    es = [e for rr in data_day.values() for e in rr["pics"].values() if e["X"]]
    if not es:
        return
    X = np.array([x for e in es for x in e["X"]])
    for name, (m, k) in models.items():
        s, i = m.predict_proba(X[:, :k])[:, 1], 0
        for e in es:
            e.setdefault("s", {})[name] = s[i:i + len(e["X"])]
            i += len(e["X"])


def pick(e, name, thr):
    if not e or not e["X"]:
        return None
    s = e["s"][name]
    j = int(np.argmax(s))
    return e["nums"][j] if s[j] >= thr else None


def points(rr, config):
    """config = ("one", name, thr) or ("v10", strip, fs): rows per time point."""
    tw, pics, rows = rr["tw"], rr["pics"], []
    taus = list(range(3 * int((tw - 9.0) // 3), -1, -3))

    def now(tau, name, thr):
        c = Counter(n for n in (pick(pics.get(round(tau + o, 1)), name, thr) for o in (0.0, 0.5, 1.0)) if n is not None)
        return next((n for n, v in c.most_common() if v >= 2), None)

    own, own_fs = {}, {}
    for tau in taus:
        if config[0] == "one":
            own[tau] = now(tau, config[1], config[2])
            lead = own[tau]
            if lead is None and own.get(tau + 3) is not None and own.get(tau + 3) == own.get(tau + 6):
                lead = own[tau + 3]
        else:
            own[tau] = now(tau, config[1], 0.5)
            lead = own[tau]
            if lead is None and own.get(tau + 3) is not None and own.get(tau + 3) == own.get(tau + 6):
                lead = own[tau + 3]
            if tau <= 36:
                own_fs[tau] = now(tau, config[2], 0.5)
                lead = own_fs[tau]
                if lead is None and own_fs.get(tau + 3) is not None and own_fs.get(tau + 3) == own_fs.get(tau + 6):
                    lead = own_fs[tau + 3]
        ref = pics.get(float(tau), {}).get("ref") if float(tau) in pics else None
        if ref is None:
            continue
        rt = tw - tau
        rows.append({"tau": tau, "part": "start" if rt < 18 else ("finish" if tau <= 36 else "middle"), "said": lead is not None,
                     "right": lead is not None and lead == ref["leader"], "lenient": lead is not None and lead in ref["leader_lenient"]})
    return rows


def summarise(rs):
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    n, said, right, len_ = len(rs), sum(r["said"] for r in rs), sum(r["right"] for r in rs), sum(r["lenient"] for r in rs)
    return {"points": n, "said": said, "right": right, "when_said": pct(right, said), "when_said_lenient": pct(len_, said),
            "right_share": pct(right, n), "wrong_share": pct(said - right, n), "right_share_lenient": pct(len_, n)}


def evaluate(data, days, config_of_day):
    """rows per (day, race) for the main-analysis races; config_of_day(day) -> config."""
    out = {}
    for d in days:
        for race, rr in data[d].items():
            if rr["agree"]:
                out[(d, race)] = points(rr, config_of_day(d))
    return out


def table(by_race):
    rows = [r for v in by_race.values() for r in v]
    return {p: summarise(rows if p == "whole race" else [r for r in rows if r["part"] == p]) for p in PARTS}


def paired(new, base, key="right_share", n=2000, seed=20261118):
    races, rng, vals = sorted(new), random.Random(seed), []
    for _ in range(n):
        pick_ = rng.choices(races, k=len(races))
        a = summarise([r for k in pick_ for r in new[k]])[key]
        b = summarise([r for k in pick_ for r in base[k]])[key]
        if a is not None and b is not None:
            vals.append(round(a - b, 1))
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def main():
    out = Path(sys.argv[1])
    data = {d: load(d) for d in DEV + HELD}
    strip = joblib.load(ROOT / "night" / "leader_fast.joblib")
    res = {"dev": {}, "held": {}, "train": {}}
    # DEV: leave the scored day out
    for d in DEV:
        rest = [x for x in DEV if x != d]
        X, y = samples(data, rest)
        res["train"][f"without_{d}"] = {"samples": int(len(y)), "positive": int(y.sum())}
        score_day(data[d], {"strip": (strip, 19), "fs": (joblib.load(ROOT / "finish2" / "models" / f"fs_leader_without_{d}.joblib"), 19),
                            "W19": (fit(X[:, :19], y), 19), "W21": (fit(X, y), 21)})
        print("scored", d, len(y), "training samples", flush=True)
    X, y = samples(data, DEV)
    res["train"]["all"] = {"samples": int(len(y)), "positive": int(y.sum())}
    w19, w21 = fit(X[:, :19], y), fit(X, y)
    (ROOT / "finish3" / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(w19, ROOT / "finish3" / "models" / "W19_all.joblib")
    joblib.dump(w21, ROOT / "finish3" / "models" / "W21_all.joblib")
    res["importance_W21"] = [round(float(v), 4) for v in w21.feature_importances_]
    fs_all = joblib.load(ROOT / "finish2" / "models" / "fs_leader_all.joblib")
    for d in HELD:
        score_day(data[d], {"strip": (strip, 19), "fs": (fs_all, 19), "W19": (w19, 19), "W21": (w21, 21)})
    configs = {"v10 (base)": ("v10", "strip", "fs"), "v9 (strip model only)": ("one", "strip", 0.5)}
    configs.update({f"{m} thr {t}": ("one", m, t) for m in ("W19", "W21") for t in THRESHOLDS})
    for key, days in (("dev", DEV), ("held", HELD)):
        runs = {name: evaluate(data, days, lambda d, c=c: c) for name, c in configs.items()}
        base = runs["v10 (base)"]
        print(f"\n== {key}: {len(base)} races, {sum(len(v) for v in base.values())} time points")
        for name, by in runs.items():
            t = table(by)
            w = t["whole race"]
            extra = {} if name == "v10 (base)" else {"right_share_minus_base": paired(by, base), "wrong_share_minus_base": paired(by, base, "wrong_share")}
            res[key][name] = {"parts": t, **extra}
            print(f"{name:24s} whole race: right share {w['right_share']}%  wrong share {w['wrong_share']}%  when said {w['when_said']}% ({w['right']}/{w['said']})"
                  f" | start {t['start']['right_share']}/{t['start']['wrong_share']}  middle {t['middle']['right_share']}/{t['middle']['wrong_share']}"
                  f"  finish {t['finish']['right_share']}/{t['finish']['wrong_share']} (when said {t['finish']['when_said']})"
                  + ("" if not extra else f" | minus base: right {extra['right_share_minus_base']} wrong {extra['wrong_share_minus_base']}"))
    # the choice, by the rule in the docstring
    base_wrong = res["dev"]["v10 (base)"]["parts"]["whole race"]["wrong_share"]
    ok = {n: v["parts"]["whole race"]["right_share"] for n, v in res["dev"].items() if n.startswith("W") and v["parts"]["whole race"]["wrong_share"] <= base_wrong}
    chosen = max(ok, key=ok.get) if ok else None
    res["chosen"] = chosen
    if chosen:
        h, hb = res["held"][chosen], res["held"]["v10 (base)"]["parts"]["whole race"]
        go = h["right_share_minus_base"][0] > 0 and h["parts"]["whole race"]["wrong_share"] - hb["wrong_share"] <= 2.0
        res["goes_to_new_days"] = bool(go)
        print(f"\nchosen on DEV: {chosen} (base wrong share on DEV {base_wrong}%). On HELD: right share {h['parts']['whole race']['right_share']}% against {hb['right_share']}%, "
              f"difference interval {h['right_share_minus_base']}; wrong share {h['parts']['whole race']['wrong_share']}% against {hb['wrong_share']}%. Goes to the new days: {go}")
    else:
        res["goes_to_new_days"] = False
        print("\nno candidate keeps the wrong share at or below the base's on DEV")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
