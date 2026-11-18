"""finish3 stage 2, confirmation on the two new days: two small changes to v10, each against v10 itself on the same pictures.
usage: g11_score2.py OUT.json DAY [DAY ...]       (CPU; needs finish3/dense/DAY.json from g4_dense_whole.py and finish2/dense/DAY.json from g10_dense_last.py)

L  leader with a lower bar for a rider's number: for the leader statement only, a rider counts when its cap probability is 0.4 or more
   (v10: 0.7). Same models (strip-trained before the last 36 s, finish-stretch model in it), same threshold 0.5, same 2-of-3 and carry rules.
   Time points: the finish and every 3 s before it back to race time 9 s. Main analysis: races whose clock agrees with the official time.
W  winner call with a guard: the number the strip-trained model names most often in the 31 pictures of the last 6 s is called only if it is
   named in 6 pictures or more (v10: 5) AND the finish-stretch model's most named number over the same pictures is the same number.
   All races run are counted (the clock-agrees subset is also given).
Verdicts (fixed in finish3/DESIGN2_KO.md before these days were processed):
  L confirmed  = "leader right share" over the whole race: new minus v10, lower end of the 95 % interval above 0; and "leader wrong share" at most 2.0 points higher
  W confirmed  = the guard calls a winner in at least 40 % of the races run, at least 90 % of its calls are right, and it makes no more wrong calls than v10's rule"""
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft
import g5_explore as g5
import g7_winner_guard as g7
import g8_explore2 as g8


def main():
    out, days = Path(sys.argv[1]), sys.argv[2:]
    for d in days:
        fc.FOLDER.setdefault(d, d)
    strip, fs_all = joblib.load(ROOT / "night" / "leader_fast.joblib"), joblib.load(ROOT / "finish2" / "models" / "fs_leader_all.joblib")
    res, runs = {"days": days, "leader": {}, "winner": {}}, {}
    for sure in (0.7, 0.4):
        ft.SURE = sure
        data = {d: g5.load(d) for d in days}
        for d in days:
            g5.score_day(data[d], {"strip": (strip, 19), "fs": (fs_all, 19), "W19": (strip, 19)})
        runs[sure] = {(d, race): g8.race_rows(rr, ("thr", 0.5)) for d in days for race, rr in data[d].items() if rr["agree"]}
        res["races_run"] = sum(len(data[d]) for d in days)
    ft.SURE = 0.7
    base, new = runs[0.7], runs[0.4]
    res["races_main"] = len(base)
    for name, by in (("v10 (bar 0.7)", base), ("bar 0.4", new)):
        t = g5.table(by)
        rows = [r for v in by.values() for r in v]
        res["leader"][name] = {"parts": t, "readable_share": round(100 * sum(r["readable"] for r in rows) / len(rows), 1),
                               "by_day": {d: g5.table({k: v for k, v in by.items() if k[0] == d})["whole race"] for d in days}}
        w = t["whole race"]
        print(f"{name:14s} {len(by)} races, {w['points']} time points: true leader readable {res['leader'][name]['readable_share']}% | whole race right share {w['right_share']}%  wrong share {w['wrong_share']}%"
              f"  when said {w['when_said']}% ({w['right']}/{w['said']}) | start {t['start']['right_share']}/{t['start']['wrong_share']}  middle {t['middle']['right_share']}/{t['middle']['wrong_share']}"
              f"  finish {t['finish']['right_share']}/{t['finish']['wrong_share']} (when said {t['finish']['when_said']})")
    dr, dw = g5.paired(new, base), g5.paired(new, base, "wrong_share")
    wb, wn = res["leader"]["v10 (bar 0.7)"]["parts"]["whole race"], res["leader"]["bar 0.4"]["parts"]["whole race"]
    okL = dr[0] > 0 and (wn["wrong_share"] - wb["wrong_share"]) <= 2.0
    res["leader"]["difference"] = {"right_share_interval": dr, "wrong_share_interval": dw, "right_share": round(wn["right_share"] - wb["right_share"], 1), "wrong_share": round(wn["wrong_share"] - wb["wrong_share"], 1)}
    res["leader"]["confirmed"] = bool(okL)
    print(f"L: right share {wn['right_share'] - wb['right_share']:+.1f} points {dr}, wrong share {wn['wrong_share'] - wb['wrong_share']:+.1f} points {dw} -> confirmed: {okL}")
    rs = [r for d in days for r in g7.races(d)]
    for name, kw in (("v10 rule (5 pictures)", {}), ("guard (6 pictures and both models agree)", {"agree": True, "need": 6})):
        res["winner"][name] = {"all": g7.tally(rs, **kw), "clock agrees": g7.tally([r for r in rs if r["agree"]], **kw),
                               "calls": [{"day": r["day"], "race": r["race"], "call": g7.call(r, **kw), "rank": r["rank"].get(g7.call(r, **kw)), "agree": r["agree"],
                                          "strip": dict(r["strip"].most_common(3)), "fs": dict(r["fs"].most_common(3))} for r in rs]}
        a, c = res["winner"][name]["all"], res["winner"][name]["clock agrees"]
        print(f"{name:42s} all {a['called']}/{a['races']} called, {a['right']} right, wrong ranks {a['wrong_ranks']} | clock agrees {c['called']}/{c['races']} called, {c['right']} right, wrong ranks {c['wrong_ranks']}")
    b, g = res["winner"]["v10 rule (5 pictures)"]["all"], res["winner"]["guard (6 pictures and both models agree)"]["all"]
    okW = g["called"] >= 0.4 * g["races"] and g["called"] > 0 and g["right"] >= 0.9 * g["called"] and (g["called"] - g["right"]) <= (b["called"] - b["right"])
    res["winner"]["confirmed"] = bool(okW)
    print(f"W: guard confirmed: {okW}")
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
