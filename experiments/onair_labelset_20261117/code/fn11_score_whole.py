"""Finish stretch 2, stage 3: score whole-race runs (fn10_whole.py) against the official passing times, by part of the race.
usage: fn11_score_whole.py OUT.json DAY [DAY ...]

Reference: fn_common.reference_full (start, S1F, corners, G3F, C4, G1F, finish; constant speed in between).
Parts: start = race time under 18 s (the strip is not on screen yet in most races; the reference is rough there because every horse
is taken to move at constant speed from the gate to the 200 m point), middle = from 18 s on until 36 s before the finish (the strip is
on screen in most of it), finish = the last 36 s (no strip), of which "finish moment" = the time point at the finish.
Main analysis: races whose clock agrees with the official time. Strict reference first, lenient (within 3 m) also given.
Intervals: bootstrap over races (2000 resamples, seed 20261118)."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn_common as fc

ROOT = fc.ROOT
PARTS = ("start", "middle", "finish", "whole race", "whole race without the finish moment")


def rows(day):
    run = json.loads((ROOT / "finish2" / "whole" / "F2" / f"{day}.json").read_text())["races"]
    off, out = fc.official(day), []
    for race, rr in sorted(run.items()):
        for p in rr.get("points", []):
            rt = rr["winner_time"] - p["to_finish"]
            ref = fc.reference_full(off[race], rt)
            if ref is None:
                continue
            said, lead = p["statements"], p.get("leader")
            out.append({"day": day, "race": race, "tau": p["to_finish"], "rt": rt, "agree": rr["clock_agrees"],
                        "part": "start" if rt < 18 else ("finish" if p["to_finish"] <= 36 else "middle"),
                        "said": len(said), "right": sum(n in ref["top3"] for n in said), "right_lenient": sum(n in ref["top3_lenient"] for n in said),
                        "lead": lead is not None, "lead_right": lead == ref["leader"], "lead_right_lenient": lead in ref["leader_lenient"]})
    return out


def summarise(rs):
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    said, led = sum(r["said"] for r in rs), [r for r in rs if r["lead"]]
    s = {"points": len(rs), "said": said, "right": sum(r["right"] for r in rs), "right_lenient": sum(r["right_lenient"] for r in rs),
         "points_good": sum(r["right"] > 0 for r in rs), "points_good_lenient": sum(r["right_lenient"] > 0 for r in rs), "points_silent": sum(r["said"] == 0 for r in rs),
         "leader_said": len(led), "leader_right": sum(r["lead_right"] for r in led), "leader_right_lenient": sum(r["lead_right_lenient"] for r in led)}
    s.update(A=pct(s["right"], said), B=pct(s["points_good"], len(rs)), A_lenient=pct(s["right_lenient"], said), B_lenient=pct(s["points_good_lenient"], len(rs)),
             leader=pct(s["leader_right"], len(led)), leader_lenient=pct(s["leader_right_lenient"], len(led)))
    return s


def part(rs, name):
    if name == "whole race":
        return rs
    if name == "whole race without the finish moment":
        return [r for r in rs if r["tau"] != 0]
    return [r for r in rs if r["part"] == name]


def interval(rs, key, n=2000, seed=20261118):
    races = sorted({(r["day"], r["race"]) for r in rs})
    by = {k: [r for r in rs if (r["day"], r["race"]) == k] for k in races}
    rng, vals = random.Random(seed), []
    for _ in range(n):
        v = summarise([x for k in rng.choices(races, k=len(races)) for x in by[k]])[key]
        if v is not None:
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None


def main():
    out, days = Path(sys.argv[1]), sys.argv[2:]
    every = [r for d in days for r in rows(d)]
    main_rows = [r for r in every if r["agree"]]
    res = {"days": days, "races": len({(r["day"], r["race"]) for r in main_rows}), "races_run": len({(r["day"], r["race"]) for r in every}), "parts": {}, "every_race_run": {}, "by_day": {}}
    for name in PARTS:
        rs = part(main_rows, name)
        s = summarise(rs)
        s["interval"] = {k: interval(rs, k) for k in ("A", "B", "leader")}
        res["parts"][name] = s
        res["every_race_run"][name] = summarise(part(every, name))
        print(f"{name:38s} points {s['points']:4d}  A {s['A']}% ({s['right']}/{s['said']}) {s['interval']['A']}  B {s['B']}% ({s['points_good']}/{s['points']}) {s['interval']['B']}"
              f"  leader {s['leader']}% ({s['leader_right']}/{s['leader_said']}) {s['interval']['leader']} | lenient A {s['A_lenient']} B {s['B_lenient']} leader {s['leader_lenient']} | silent {s['points_silent']}")
    for d in days:
        res["by_day"][d] = {name: {k: v for k, v in summarise(part([r for r in main_rows if r["day"] == d], name)).items() if k in ("points", "said", "right", "A", "B", "leader", "leader_said", "leader_right")} for name in PARTS}
        print(" ", d, "  ".join(f"{name}: A {v['A']} B {v['B']}" for name, v in res["by_day"][d].items()))
    print("  every race run (" + str(res["races_run"]) + "): " + "  ".join(f"{k}: A {v['A']} B {v['B']}" for k, v in res["every_race_run"].items()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
