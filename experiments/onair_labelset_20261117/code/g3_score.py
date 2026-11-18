"""finish3: score whole-race runs (g2_whole_v10.py) against the official passing times.
usage: g3_score.py OUT.json NAME DAY [DAY ...]            NAME = folder under finish3/whole (V10, or a stage-2 candidate)

Reference and parts as fn11_score_whole.py (fn_common.reference_full; start = race time under 18 s, finish = the last 36 s, middle = between).
Main analysis: races whose clock agrees with the official time. Strict reference; lenient (within 3 m) also given.
Counted:
  A, B                 stated numbers in the reference's first three / all stated; time points with a right number / all time points
  leader when said     leader statements naming the reference's first / all leader statements
  leader right share   time points whose leader statement names the reference's first / ALL time points       (headline, higher is better)
  leader wrong share   time points whose leader statement names another horse / ALL time points              (headline, lower is better)
  the same four for "base" = the strip-trained leader model on the same time points (v9's leader)
  winner call          races with a call, calls naming the official winner; with the finish from the official time and from the clock stop
Intervals: bootstrap over races (2000 resamples, seed 20261118); paired differences new minus base on the same resampled races."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn_common as fc

ROOT = fc.ROOT
PARTS = ("start", "middle", "finish", "whole race", "whole race without the finish moment")


def rows(name, day):
    run = json.loads((ROOT / "finish3" / "whole" / name / f"{day}.json").read_text())["races"]
    off, out, races = fc.official(day), [], []
    for race, rr in sorted(run.items()):
        rank = {h["number"]: h["rank"] for h in off[race]["horses"]}
        w = {k: v["number"] for k, v in rr.get("winner_call", {}).items()}
        races.append({"day": day, "race": race, "agree": rr["clock_agrees"], "winner": next((n for n, r in rank.items() if r == 1), None),
                      **{f"call_{k}": v for k, v in w.items()}, **{f"rank_{k}": rank.get(v) for k, v in w.items()}})
        for p in rr.get("points", []):
            rt = rr["winner_time"] - p["to_finish"]
            ref = fc.reference_full(off[race], rt)
            if ref is None:
                continue
            said, lead, base = p["statements"], p.get("leader"), p.get("leader_strip_model")
            out.append({"day": day, "race": race, "tau": p["to_finish"], "rt": rt, "agree": rr["clock_agrees"],
                        "part": "start" if rt < 18 else ("finish" if p["to_finish"] <= 36 else "middle"),
                        "said": len(said), "right": sum(n in ref["top3"] for n in said), "right_lenient": sum(n in ref["top3_lenient"] for n in said),
                        "lead": lead is not None, "lead_right": lead is not None and lead == ref["leader"], "lead_right_lenient": lead is not None and lead in ref["leader_lenient"],
                        "base": base is not None, "base_right": base is not None and base == ref["leader"], "base_right_lenient": base is not None and base in ref["leader_lenient"]})
    return out, races


def summarise(rs):
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    n, said = len(rs), sum(r["said"] for r in rs)
    s = {"points": n, "said": said, "right": sum(r["right"] for r in rs), "right_lenient": sum(r["right_lenient"] for r in rs),
         "points_good": sum(r["right"] > 0 for r in rs), "points_silent": sum(r["said"] == 0 for r in rs)}
    s.update(A=pct(s["right"], said), B=pct(s["points_good"], n), A_lenient=pct(s["right_lenient"], said))
    for k in ("lead", "base"):
        m, ok, okl = sum(r[k] for r in rs), sum(r[f"{k}_right"] for r in rs), sum(r[f"{k}_right_lenient"] for r in rs)
        s.update({f"{k}_said": m, f"{k}_right": ok, f"{k}_wrong": m - ok, f"{k}_when_said": pct(ok, m), f"{k}_when_said_lenient": pct(okl, m),
                  f"{k}_right_share": pct(ok, n), f"{k}_wrong_share": pct(m - ok, n), f"{k}_right_share_lenient": pct(okl, n)})
    return s


def part(rs, name):
    if name == "whole race":
        return rs
    if name == "whole race without the finish moment":
        return [r for r in rs if r["tau"] != 0]
    return [r for r in rs if r["part"] == name]


KEYS = ("A", "B", "lead_when_said", "lead_right_share", "lead_wrong_share", "base_when_said", "base_right_share", "base_wrong_share")
DIFFS = (("lead_when_said", "base_when_said"), ("lead_right_share", "base_right_share"), ("lead_wrong_share", "base_wrong_share"))


def intervals(rs, n=2000, seed=20261118):
    races = sorted({(r["day"], r["race"]) for r in rs})
    by = {k: [r for r in rs if (r["day"], r["race"]) == k] for k in races}
    rng = random.Random(seed)
    vals = {k: [] for k in KEYS + tuple(f"{a}_minus_{b}" for a, b in DIFFS)}
    for _ in range(n):
        s = summarise([x for k in rng.choices(races, k=len(races)) for x in by[k]])
        for k in KEYS:
            if s[k] is not None:
                vals[k].append(s[k])
        for a, b in DIFFS:
            if s[a] is not None and s[b] is not None:
                vals[f"{a}_minus_{b}"].append(round(s[a] - s[b], 1))
    out = {}
    for k, v in vals.items():
        v.sort()
        out[k] = [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if v else None
    return out


def winners(races, key):
    called = [r for r in races if r.get(f"call_{key}") is not None]
    return {"races": len(races), "called": len(called), "right": sum(r[f"rank_{key}"] == 1 for r in called),
            "rank_of_called": sorted(str(r[f"rank_{key}"]) for r in called)}


def main():
    out, name, days = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    every, races = [], []
    for d in days:
        a, b = rows(name, d)
        every += a
        races += b
    main_rows, main_races = [r for r in every if r["agree"]], [r for r in races if r["agree"]]
    res = {"name": name, "days": days, "races": len(main_races), "races_run": len(races), "parts": {}, "every_race_run": {}, "by_day": {}, "winner": {}, "race_rows": races}
    for nm in PARTS:
        rs = part(main_rows, nm)
        s = summarise(rs)
        s["interval"] = intervals(rs)
        res["parts"][nm] = s
        res["every_race_run"][nm] = summarise(part(every, nm))
        iv = s["interval"]
        print(f"{nm:38s} points {s['points']:4d}  A {s['A']}% ({s['right']}/{s['said']}) {iv['A']}  B {s['B']}% ({s['points_good']}/{s['points']}) {iv['B']}")
        print(f"{'':38s} leader when said {s['lead_when_said']}% ({s['lead_right']}/{s['lead_said']}) {iv['lead_when_said']} lenient {s['lead_when_said_lenient']}%"
              f" | right share {s['lead_right_share']}% {iv['lead_right_share']}  wrong share {s['lead_wrong_share']}% {iv['lead_wrong_share']}")
        print(f"{'':38s} base (strip-trained leader): when said {s['base_when_said']}% ({s['base_right']}/{s['base_said']})  right share {s['base_right_share']}%  wrong share {s['base_wrong_share']}%"
              f" | new minus base: when said {iv['lead_when_said_minus_base_when_said']}  right share {iv['lead_right_share_minus_base_right_share']}  wrong share {iv['lead_wrong_share_minus_base_wrong_share']}")
    for d in days:
        res["by_day"][d] = {nm: {k: v for k, v in summarise(part([r for r in main_rows if r["day"] == d], nm)).items()
                                 if k in ("points", "A", "B", "lead_when_said", "lead_said", "lead_right", "lead_right_share", "lead_wrong_share", "base_when_said", "base_right_share")} for nm in PARTS}
        v = res["by_day"][d]
        print(" ", d, "whole race A", v["whole race"]["A"], "B", v["whole race"]["B"], "| finish part leader when said", v["finish"]["lead_when_said"], f"({v['finish']['lead_right']}/{v['finish']['lead_said']})",
              "| whole race leader right share", v["whole race"]["lead_right_share"], "wrong share", v["whole race"]["lead_wrong_share"])
    for key in ("official_anchor", "clock_stop"):
        res["winner"][key] = {"main": winners(main_races, key), "every_race_run": winners(races, key)}
        m, e = res["winner"][key]["main"], res["winner"][key]["every_race_run"]
        print(f"winner call ({key}): main {m['called']}/{m['races']} called, {m['right']} right, ranks {m['rank_of_called']} | every race run {e['called']}/{e['races']} called, {e['right']} right")
    w = res["every_race_run"]["whole race"]
    print(f"every race run ({res['races_run']}): whole race A {w['A']} B {w['B']} leader right share {w['lead_right_share']} wrong share {w['lead_wrong_share']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
