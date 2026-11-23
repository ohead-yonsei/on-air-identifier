"""v12 stage 3: replay the program's statement rule on the whole-race dense pictures, for v11 and for candidate v12 rules, scored on the
official order. usage: v12_rule.py OUT.json MODELS(lodo|all) DAY [DAY ...]      (CPU)

Time points are every 3 s back from the winner's finish down to race time 9 s (as bench2/whole); each uses the pictures at t, t-0.5 and
t-1.0 s, as the program does. MODELS=lodo: a development day is scored with the v12 models learned without it; all: the models of all
development days (for days outside the development set).
  v11   the shipped rule (F2): per picture the two riders with the highest "first three" score at 0.6 or more; stated when picked in 2 of 3
        pictures; carried when stated at t-3 and t-6 s; fallback when nothing is stated. Scored here against the first five as well.
  v12   per picture the K riders with the highest "first five" score at tau or more; stated when picked in `need` of 3 pictures; optional
        carry as v11; ordered by the mean predicted metres behind the leader over the pictures where the rider is sure (a carried
        horse keeps its last value). Two named horses are said to be in order when their predicted gap differs by delta metres or more,
        otherwise "side by side".
Scores per time point: horses named; named horses in the first five (strict) / not more than 3 m behind the fifth (lenient); points with a
wrong number; pair orders claimed and right (strict: official order; lenient: wrong by 3 m or less counts as right)."""
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft
from v12_train import DEV, truth

V11_MODEL = ROOT / "night" / "l1_fast.joblib"
TAUS, KS, NEEDS, DELTAS = (0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96), (3, 4, 5), (2, 3), (0.0, 1.0, 2.0, 3.0, 4.0)
START_SECONDS = 18.0      # start_v11: before this race time the group is v11's statements (put in order by the gap model)
MARGIN = fc.MARGIN_M


def part(r, tf):
    return "start" if r < 18.0 else "finish" if tf <= 36.0 else "middle"


def race_points(dr, off_race, m11, m5, mg):
    """-> list of time points, each {r, tf, truth, per picture: [{num: (p3, p5, g)} ...]} for the 3 pictures (newest first)."""
    pics = dr["pictures"]
    whos = ft.identify(pics, [[0] * len(p["boxes"]) for p in pics], memory=False)
    by_k = {}
    for p, w in zip(pics, whos):
        k = int(round(p["to_finish"] / 0.5))
        rec = {}
        idx = [i for i, (num, c) in enumerate(w) if num and c >= ft.SURE]
        if idx and p["boxes"]:
            F = np.array(ft.features(p["boxes"], [c for _, c in w]))[idx]
            p3, p5, g = m11.predict_proba(F)[:, 1], m5.predict_proba(F)[:, 1], mg.predict(F)
            for j, i in enumerate(idx):
                rec[w[i][0]] = (float(p3[j]), float(p5[j]), float(g[j]))
        by_k[k] = rec
    out, j = [], 0
    while True:
        tf = 3.0 * j
        r = dr["winner_time"] - tf
        if r < 9.0:
            break
        ks = [6 * j, 6 * j + 1, 6 * j + 2]
        tr = truth(off_race, r)
        if tr is not None and all(k in by_k for k in ks):
            out.append({"r": r, "tf": tf, "truth": tr, "pics": [by_k[k] for k in ks]})
        j += 1
    return out[::-1]          # oldest first, so that carry sees earlier points


def run_v11(points):
    hist, res = {}, []
    for i, pt in enumerate(points):
        votes, top = Counter(), {}
        for rec in pt["pics"]:
            ranked = sorted(rec, key=lambda n: -rec[n][0])[:ft.K]
            for n in ranked:
                if rec[n][0] >= ft.TAU:
                    votes[n] += 1
            for n, v in rec.items():
                top[n] = max(top.get(n, 0.0), v[0])
        now = {n for n, v in votes.items() if v >= 2}
        hist[i] = now
        prev = [hist.get(i - 1, set()), hist.get(i - 2, set())]
        carried = (prev[0] & prev[1]) - now
        added = set()
        if not (now | carried):
            added = prev[0] & set(top)
            if not added and top:
                n = max(top, key=top.get)
                if top[n] >= ft.FALLBACK_SINGLE:
                    added = {n}
        res.append(sorted(now | carried | added))
    return res


def run_v12(points, tau, K, need=2, carry=True, start_v11=None):
    """start_v11: the v11 statements of the same points; when given, points before START_SECONDS of race time name those instead."""
    hist, gap_hist, res = {}, {}, []
    for i, pt in enumerate(points):
        votes, gaps = Counter(), {}
        for rec in pt["pics"]:
            for n in sorted(rec, key=lambda n: -rec[n][1])[:K]:
                if rec[n][1] >= tau:
                    votes[n] += 1
            for n, v in rec.items():
                gaps.setdefault(n, []).append(v[2])
        now = {n for n, v in votes.items() if v >= need}
        hist[i] = now
        carried = ((hist.get(i - 1, set()) & hist.get(i - 2, set())) - now) if carry else set()
        named = now | carried
        if start_v11 is not None and pt["r"] < START_SECONDS:
            named = set(start_v11[i])
        g = {}
        for n in named:
            g[n] = float(np.mean(gaps[n])) if n in gaps else gap_hist.get(n, 99.0)
        gap_hist.update(g)
        res.append([(n, round(g[n], 2)) for n in sorted(named, key=lambda n: g[n])])
    return res


def score_names(named, tr):
    s = {"named": len(named), "right": 0, "right_lenient": 0, "wrong_point": 0, "wrong_point_lenient": 0}
    for n in named:
        rk, b = tr["rank"](n), tr["behind"](n)
        s["right"] += int(rk is not None and rk <= 5)
        s["right_lenient"] += int(b is not None and b <= tr["fifth"] + MARGIN)
    s["wrong_point"] = int(s["right"] < len(named))
    s["wrong_point_lenient"] = int(s["right_lenient"] < len(named))
    return s


def score_pairs(ordered, tr, delta):
    s = {"pairs": 0, "pairs_right": 0, "pairs_right_lenient": 0, "side_by_side": 0, "side_by_side_within_3m": 0}
    for (a, ga), (b, gb) in itertools.combinations(ordered, 2):
        ba, bb = tr["behind"](a), tr["behind"](b)
        if ba is None or bb is None:
            continue
        if gb - ga >= delta:
            s["pairs"] += 1
            s["pairs_right"] += int(ba < bb)
            s["pairs_right_lenient"] += int(ba <= bb + MARGIN)
        else:
            s["side_by_side"] += 1
            s["side_by_side_within_3m"] += int(abs(ba - bb) <= MARGIN)
    return s


def add(acc, key, s):
    a = acc.setdefault(key, Counter())
    a.update(s)
    a["points"] += 1


def summary(c):
    p = c["points"] or 1
    pc = lambda a, b: round(100 * a / b, 2) if b else None
    return {"points": c["points"], "named_per_point": round(c["named"] / p, 3), "right_per_point": round(c["right"] / p, 3),
            "right_lenient_per_point": round(c["right_lenient"] / p, 3), "named_right_%": pc(c["right"], c["named"]),
            "wrong_point_%": pc(c["wrong_point"], p), "wrong_point_lenient_%": pc(c["wrong_point_lenient"], p),
            "pairs_per_point": round(c["pairs"] / p, 3) if "pairs" in c else None, "pairs_right_%": pc(c["pairs_right"], c["pairs"]),
            "pairs_right_lenient_%": pc(c["pairs_right_lenient"], c["pairs"]), "side_by_side_per_point": round(c["side_by_side"] / p, 3) if "side_by_side" in c else None,
            "side_by_side_within_3m_%": pc(c["side_by_side_within_3m"], c["side_by_side"])}


def main():
    out_path, mode, days = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    m11 = joblib.load(V11_MODEL)
    acc = {}
    for day in days:
        tag = f"without_{day}" if mode == "lodo" else "all"
        assert mode == "all" or day in DEV
        m5, mg = joblib.load(ROOT / "v12" / "models" / f"top5_{tag}.joblib"), joblib.load(ROOT / "v12" / "models" / f"gap_{tag}.joblib")
        dense = json.loads((ROOT / "v12" / "dense" / f"{day}.json").read_text())["races"]
        off = fc.official(day)
        for race, dr in sorted(dense.items()):
            if not dr["clock_agrees"]:
                continue
            pts = race_points(dr, off[race], m11, m5, mg)
            v11 = run_v11(pts)
            for pt, named in zip(pts, v11):
                for prt in ("all", part(pt["r"], pt["tf"])):
                    add(acc, ("v11", prt), score_names(named, pt["truth"]))
            for tau, K, need, carry, sv in itertools.product(TAUS, KS, NEEDS, (True, False), (False, True)):
                res = run_v12(pts, tau, K, need=need, carry=carry, start_v11=v11 if sv else None)
                for pt, ordered in zip(pts, res):
                    sn = score_names([n for n, _ in ordered], pt["truth"])
                    for delta in DELTAS:
                        s = Counter(sn)
                        s.update(score_pairs(ordered, pt["truth"], delta))
                        for prt in ("all", part(pt["r"], pt["tf"])):
                            add(acc, (f"v12 tau={tau} K={K} need={need} carry={int(carry)} start_v11={int(sv)} delta={delta}", prt), s)
        print(day, "done", flush=True)
    rows = {}
    for (name, prt), c in acc.items():
        rows.setdefault(name, {})[prt] = summary(c)
    out_path.write_text(json.dumps({"days": days, "models": mode, "rows": rows}, indent=1))
    base = rows["v11"]["all"]
    print("v11", base)
    best = sorted((r["all"]["right_per_point"], n) for n, r in rows.items() if n != "v11" and r["all"]["wrong_point_%"] is not None and r["all"]["wrong_point_%"] <= 5.0)
    for v, n in best[-12:]:
        print(n, rows[n]["all"])


if __name__ == "__main__":
    main()
