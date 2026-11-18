"""Finish stretch 2, stage 2: calling the winner from the dense pictures of the last seconds.
usage: fn9_winner.py explore DAY [DAY ...]            grid of rules on development days (prints a table, writes finish2/explore/winner_grid.json)
       fn9_winner.py call RULE OUT.json DAY [DAY ...]  one rule; RULE = memory:models:span:need:ratio   e.g. P:strip:6.0:5:1.0,  or 3s

A rule: in the pictures of the last `span` seconds before the finish (5 a second), each picture names a leader or not
(fn_tracks.decide: the sure rider with the highest leader score, if 0.5 or more). The number named most often is called the winner
when it was named in at least `need` pictures and at least `ratio` times as often as the runner-up. Otherwise no call.
memory: P = each picture on its own, M = riders followed across pictures.  models: strip = the strip-trained fast models,
fs = models learned on the finish stretch from official times (the model that left the day out on development days).
The answer is the official winner. Main analysis: races whose clock agrees with the official time."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft
from fn7_method import models

_cache = {}


def picks(day, memory, mod):
    """race -> list of (to_finish, leader pick or None, {number: leader score})"""
    key = (day, memory, mod)
    if key in _cache:
        return _cache[key]
    order_model, leader_model = models("M6f" if mod == "fs" else "M6", day)
    dense = json.loads((ROOT / "finish2" / "dense" / f"{day}.json").read_text())["races"]
    out = {}
    for race, dr in dense.items():
        pics = dr["pictures"]
        who = ft.identify(pics, ft.link(pics), memory=(memory == "M"))
        ds = [ft.decide(p, w, order_model, leader_model) for p, w in zip(pics, who)]
        out[race] = {"agree": dr["clock_agrees"], "picks": [(p["to_finish"], d["pick"], d["first"]) for p, d in zip(pics, ds)]}
    _cache[key] = out
    return out


def call(pk, span, need, ratio, after=0.0):
    c = Counter(n for t, n, _ in pk if n is not None and -after - 0.05 <= t <= span + 0.05)
    if not c:
        return None, c
    (a, na), rest = c.most_common(1)[0], c.most_common(2)[1:]
    nb = rest[0][1] if rest else 0
    return (a if na >= need and na >= ratio * nb else None), c


def evaluate(days, memory, mod, span, need, ratio, after=0.0, every=False):
    called = right = races = 0
    rows = []
    for day in days:
        off = fc.official(day)
        for race, v in sorted(picks(day, memory, mod).items()):
            if not (v["agree"] or every):
                continue
            rank = {h["number"]: h["rank"] for h in off[race]["horses"]}
            w, c = call(v["picks"], span, need, ratio, after)
            races += 1
            called += w is not None
            right += w is not None and rank.get(w) == 1
            rows.append({"day": day, "race": race, "call": w, "rank_of_call": rank.get(w), "winner": next((n for n, r in rank.items() if r == 1), None),
                         "counts": dict(c.most_common(4)), "clock_agrees": v["agree"]})
    return {"races": races, "called": called, "right": right, "rows": rows}


def main():
    if sys.argv[1] == "explore":
        days, grid = sys.argv[2:], []
        for memory in "PM":
            for mod in ("strip", "fs"):
                if mod == "fs" and not (ROOT / "finish2" / "models" / "fs_order_all.joblib").exists():
                    continue
                for span in (1.0, 2.0, 3.0, 4.0, 6.0):
                    for need in (2, 3, 5):
                        for ratio in (1.0, 2.0, 3.0):
                            e = evaluate(days, memory, mod, span, need, ratio)
                            grid.append({"memory": memory, "models": mod, "span": span, "need": need, "ratio": ratio, **{k: e[k] for k in ("races", "called", "right")}})
        grid.sort(key=lambda g: (-(g["right"] / max(g["called"], 1) >= 0.9), -g["right"]))
        for g in grid[:40]:
            print(f"{g['memory']} {g['models']:5s} span {g['span']} need {g['need']} ratio {g['ratio']}: called {g['called']}/{g['races']}, right {g['right']} ({100 * g['right'] / max(g['called'], 1):.1f}%)")
        (ROOT / "finish2" / "explore").mkdir(parents=True, exist_ok=True)
        (ROOT / "finish2" / "explore" / "winner_grid.json").write_text(json.dumps(grid, indent=1))
    elif sys.argv[2] == "3s":           # comparison rule on the 3 s output: the leader stated at the finish, else the one stated 3 s before
        days, res = sys.argv[4:], {"rule": "3s", "days": sys.argv[4:]}
        for name, every in (("main", False), ("every_race_run", True)):
            rows = []
            for day in days:
                off, tms = fc.official(day), fc.times(day)
                for race, rr in sorted(json.loads((ROOT / "finish2" / "runs" / "F2" / f"{day}.json").read_text())["races"].items()):
                    if "points" not in rr or not (fc.video_finish(tms.get(race), rr["winner_time"])[2] or every):
                        continue
                    lead = {p["to_finish"]: p.get("leader") for p in rr["points"]}
                    w = lead.get(0) if lead.get(0) is not None else lead.get(3)
                    rank = {h["number"]: h["rank"] for h in off[race]["horses"]}
                    rows.append({"day": day, "race": race, "call": w, "rank_of_call": rank.get(w), "winner": next((n for n, r in rank.items() if r == 1), None)})
            e = {"races": len(rows), "called": sum(r["call"] is not None for r in rows), "right": sum(r["call"] is not None and r["rank_of_call"] == 1 for r in rows), "rows": rows}
            res[name] = e
            print(name, f"called {e['called']}/{e['races']}, right {e['right']}" + (f" ({100 * e['right'] / e['called']:.1f}%)" if e["called"] else ""))
        Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
    else:
        memory, mod, span, need, ratio = sys.argv[2].split(":")
        days = sys.argv[4:]
        res = {"rule": sys.argv[2], "days": days}
        for name, every in (("main", False), ("every_race_run", True)):
            e = evaluate(days, memory, mod, float(span), int(need), float(ratio), every=every)
            res[name] = e
            print(name, f"called {e['called']}/{e['races']}, right {e['right']}" + (f" ({100 * e['right'] / e['called']:.1f}%)" if e["called"] else ""),
                  "| rank of the called horse:", dict(sorted(Counter(str(r["rank_of_call"]) for r in e["rows"] if r["call"] is not None).items())))
        Path(sys.argv[3]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
