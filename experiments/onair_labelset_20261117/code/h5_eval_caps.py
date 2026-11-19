"""caps2, step 4: what a cap classifier does to the identifier, on the same riders (same boxes), against the official order.
usage: h5_eval_caps.py OUT.json NEW_NAME DAY [DAY ...]         NEW_NAME = folder under caps2/dense (h4_reprob.py); compared with the current classifier

For both classifiers, on the races whose clock agrees with the official time, time points as fn10_whole.py (the finish and every 3 s before it
back to race time 9 s), everything else as v10 (strip-trained order model; leader from the strip-trained model, in the last 36 s from the
finish-stretch model; threshold 0.5; 2 of 3 pictures; carry):
  readable            the true leader's number is a sure rider (0.7 or more) in at least 2 of the 3 pictures / all time points
  leader right/wrong  time points whose leader statement names the true first / another horse, over all time points (also with the leader
                      bar 0.4, the option --leader-sure)
  A, B                stated numbers that are in the true first three / all stated; time points with a right number / all time points
  the first three for the time points whose true leader is number 7, 10 or 11 and above
  winner call         v10's rule on the pictures of the last 6 s (all races run)
Differences new minus current with a bootstrap over races (2000 resamples, seed 20261118)."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft
import g5_explore as g5
import g8_explore2 as g8

ORDER = joblib.load(ROOT / "night" / "l1_fast.joblib")
STRIP = joblib.load(ROOT / "night" / "leader_fast.joblib")
FS = joblib.load(ROOT / "finish2" / "models" / "fs_leader_all.joblib")
WEAK = (7, 10, 11)


def path(name, kind, day):
    if name == "current":
        return ROOT / ("finish3" if kind == "whole" else "finish2") / "dense" / f"{day}.json"
    return ROOT / "caps2" / "dense" / name / kind / f"{day}.json"


def load(name, day, sure):
    """as g5_explore.load, from the given classifier's file"""
    ft.SURE = sure
    dense = json.loads(path(name, "whole", day).read_text())["races"]
    off, out = fc.official(day), {}
    for race, dr in sorted(dense.items()):
        pics = {}
        for p in dr["pictures"]:
            e = {"nums": [], "X": [], "ref": fc.reference_full(off[race], dr["winner_time"] - p["to_finish"])}
            if p["boxes"]:
                who = ft.identify([p], [list(range(len(p["boxes"])))], memory=False)[0]
                F = ft.features(p["boxes"], [c for _, c in who])
                for r, (num, c) in enumerate(who):
                    if num and c >= sure:
                        e["nums"].append(num)
                        e["X"].append([float(v) for v in F[r]])
            pics[p["to_finish"]] = e
        out[race] = {"agree": dr["clock_agrees"], "tw": dr["winner_time"], "pics": pics, "raw": dr}
    ft.SURE = 0.7
    return out


def leader_rows(name, day, sure):
    data = load(name, day, sure)
    g5.score_day(data, {"strip": (STRIP, 19), "fs": (FS, 19), "W19": (STRIP, 19)})
    out = {}
    for race, rr in data.items():
        if not rr["agree"]:
            continue
        rows = g8.race_rows(rr, ("thr", 0.5))
        for r in rows:
            r["true"] = min(rr["pics"][float(r["tau"])]["ref"]["leader"], 11)
        out[(day, race)] = rows
    return out, data


def statement_rows(data, day):
    """A/B of the number statements (v9's rule) on the dense pictures."""
    out = {}
    for race, rr in data.items():
        if not rr["agree"]:
            continue
        per = {}
        for p in rr["raw"]["pictures"]:
            if p["boxes"]:
                who = ft.identify([p], [list(range(len(p["boxes"])))], memory=False)[0]
                per[round(p["to_finish"], 1)] = ft.decide(p, who, ORDER, STRIP)
            else:
                per[round(p["to_finish"], 1)] = {"taken": set(), "lead3": {}, "first": {}, "pick": None, "riders": 0}
        taus = list(range(3 * int((rr["tw"] - 9.0) // 3), -1, -3))
        rows = []
        for pt in ft.statements(per, taus, {}, {}, offsets=(0.0, 0.5, 1.0), need=2):
            ref = rr["pics"].get(float(pt["to_finish"]), {}).get("ref")
            if ref is None:
                continue
            said = pt["statements"]
            rows.append({"said": len(said), "right": sum(n in ref["top3"] for n in said)})
        out[(day, race)] = rows
    return out


def winner(name, day):
    f = path(name, "last", day)
    if not f.exists():
        return []
    off, out = fc.official(day), []
    for race, dr in sorted(json.loads(f.read_text())["races"].items()):
        pics = dr["pictures"]
        who = ft.identify(pics, ft.link(pics), memory=False)
        c = Counter()
        for p, w in zip(pics, who):
            if -0.05 <= p["to_finish"] <= 6.05:
                n = ft.decide(p, w, ORDER, STRIP)["pick"]
                if n is not None:
                    c[n] += 1
        top = c.most_common(1)
        call = top[0][0] if top and top[0][1] >= 5 else None
        rank = {h["number"]: h["rank"] for h in off[race]["horses"]}
        out.append({"day": day, "race": race, "call": call, "rank": rank.get(call), "winner": next((n for n, r in rank.items() if r == 1), None)})
    return out


def lead_sum(rows):
    n = len(rows)
    pct = lambda a: round(100 * a / n, 1) if n else None
    return {"points": n, "readable": pct(sum(r["readable"] for r in rows)), "right": pct(sum(r["right"] for r in rows)), "wrong": pct(sum(r["said"] and not r["right"] for r in rows))}


def ab_sum(rows):
    said, right, n = sum(r["said"] for r in rows), sum(r["right"] for r in rows), len(rows)
    return {"A": round(100 * right / said, 1) if said else None, "B": round(100 * sum(r["right"] > 0 for r in rows) / n, 1) if n else None, "said": said, "right": right, "points": n}


def diff(new, old, f, key, n=2000, seed=20261118):
    races, rng, vals = sorted(new), random.Random(seed), []
    for _ in range(n):
        pk = rng.choices(races, k=len(races))
        a, b = f([r for k in pk for r in new[k]])[key], f([r for k in pk for r in old[k]])[key]
        if a is not None and b is not None:
            vals.append(round(a - b, 1))
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None


def main():
    out, new_name, days = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    for d in days:
        fc.FOLDER.setdefault(d, d)
    R = {}
    for name in ("current", new_name):
        lead7, lead4, ab, win = {}, {}, {}, []
        for d in days:
            a, data = leader_rows(name, d, 0.7)
            lead7.update(a)
            ab.update(statement_rows(data, d))
            b, _ = leader_rows(name, d, 0.4)
            # "readable" is defined with the 0.7 bar: copy it over
            for k in b:
                for r4, r7 in zip(b[k], a[k]):
                    r4["readable"] = r7["readable"]
            lead4.update(b)
            win += winner(name, d)
        R[name] = {"lead7": lead7, "lead4": lead4, "ab": ab, "win": win}
    res = {"days": days, "new": new_name, "races": len(R["current"]["lead7"])}
    flat = lambda by: [r for v in by.values() for r in v]
    weak = lambda by: {k: [r for r in v if r["true"] in WEAK] for k, v in by.items()}
    for name in ("current", new_name):
        r = R[name]
        called = [w for w in r["win"] if w["call"] is not None]
        res[name] = {"leader_bar_0.7": lead_sum(flat(r["lead7"])), "leader_bar_0.4": lead_sum(flat(r["lead4"])), "numbers": ab_sum(flat(r["ab"])),
                     "true_leader_7_10_11+": lead_sum(flat(weak(r["lead7"]))), "true_leader_other": lead_sum([x for x in flat(r["lead7"]) if x["true"] not in WEAK]),
                     "winner": {"races": len(r["win"]), "called": len(called), "right": sum(w["rank"] == 1 for w in called), "wrong_ranks": sorted(w["rank"] or 99 for w in called if w["rank"] != 1)}}
        v = res[name]
        print(f"{name:14s} {res['races']} races, {v['leader_bar_0.7']['points']} time points | readable {v['leader_bar_0.7']['readable']}% | leader right {v['leader_bar_0.7']['right']}% wrong {v['leader_bar_0.7']['wrong']}%"
              f" | with bar 0.4: right {v['leader_bar_0.4']['right']}% wrong {v['leader_bar_0.4']['wrong']}% | numbers A {v['numbers']['A']}% ({v['numbers']['right']}/{v['numbers']['said']}) B {v['numbers']['B']}%")
        print(f"{'':14s} true leader 7/10/11+: {v['true_leader_7_10_11+']} | other: {v['true_leader_other']} | winner call {v['winner']}")
    n, o = R[new_name], R["current"]
    res["difference"] = {"readable": diff(n["lead7"], o["lead7"], lead_sum, "readable"), "leader_right": diff(n["lead7"], o["lead7"], lead_sum, "right"),
                         "leader_wrong": diff(n["lead7"], o["lead7"], lead_sum, "wrong"), "leader_right_bar_0.4": diff(n["lead4"], o["lead4"], lead_sum, "right"),
                         "leader_wrong_bar_0.4": diff(n["lead4"], o["lead4"], lead_sum, "wrong"), "A": diff(n["ab"], o["ab"], ab_sum, "A"), "B": diff(n["ab"], o["ab"], ab_sum, "B"),
                         "readable_7_10_11+": diff(weak(n["lead7"]), weak(o["lead7"]), lead_sum, "readable"), "leader_right_7_10_11+": diff(weak(n["lead7"]), weak(o["lead7"]), lead_sum, "right")}
    print("new minus current (95 % intervals):", json.dumps(res["difference"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
