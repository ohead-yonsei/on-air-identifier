"""Finish stretch 2, step 4: score identifier runs on the last 36 s against the official passing times.
usage: fn4_score.py OUT.json CONFIG[=VARIANT] DAY [DAY ...]     e.g.  fn4_score.py finish2/score_M.json F2 L1 -- otherday6 otherday7 otherday8
       (configs before "--", days after)

Per time point (36, 33, ..., 0 s before the winner finishes): the stated numbers and the stated leader are compared with the order
computed from the official passing times at that race time (fn_common.reference).
Main analysis: races whose on-screen clock agrees with the official winning time (fn_common.video_finish); all races run are also reported.
A = stated numbers that are in the first three / all stated numbers.   B = time points with at least one right number / all time points.
Leader = stated leaders that are first / all stated leaders.   "lenient" also accepts a horse within 3 m of the third (of the first).
Windows: far = more than 12 s to go, last = the last 12 s. Intervals: bootstrap over races (2000 resamples, seed 20261118).
Variants (computed from the same run): F2 as run; F1s = F2 without the fallback statements."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn_common as fc

ROOT = fc.ROOT


def rows(config, day):
    """One row per scored time point."""
    base, variant = (config.split("=") + [None])[:2] if "=" in config else (config, None)
    run = json.loads((ROOT / "finish2" / "runs" / base / f"{day}.json").read_text())
    off = fc.official(day)
    out = []
    tms = fc.times(day)
    for race, rr in sorted(run["races"].items()):
        if "points" not in rr:
            continue
        agree = fc.video_finish(tms.get(race), rr["winner_time"])[2]
        for p in rr["points"]:
            tau = p["to_finish"]
            if tau > 36:
                continue
            ref = fc.reference(off[race], rr["winner_time"] - tau)
            if ref is None:
                continue
            said = p["statements"]
            if variant == "nofallback":
                said = sorted(d["number"] for d in p["detail"] if d["source"] != "fallback")
            out.append({"day": day, "race": race, "tau": tau, "clock_agrees": agree, "said": said, "leader": p.get("leader"),
                        "right": [n for n in said if n in ref["top3"]], "right_lenient": [n for n in said if n in ref["top3_lenient"]],
                        "leader_right": p.get("leader") == ref["leader"], "leader_right_lenient": p.get("leader") in ref["leader_lenient"],
                        "leader_rank": (ref["order"].index(p["leader"]) + 1) if p.get("leader") in ref["order"] else None,
                        "top3": ref["top3"], "lenient_size": len(ref["top3_lenient"]), "front_size": len(ref["leader_lenient"]),
                        "winner": next((h["number"] for h in off[race]["horses"] if h.get("rank") == 1), None), "sources": {d["number"]: d["source"] for d in p["detail"]},
                        "riders": [x["riders"] for x in p["pictures"]], "detection": [x["detection"] for x in p["pictures"]]})
    return out


def summarise(rs):
    n_said = sum(len(r["said"]) for r in rs)
    led = [r for r in rs if r["leader"] is not None]
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    s = {"points": len(rs), "said": n_said, "right": sum(len(r["right"]) for r in rs), "right_lenient": sum(len(r["right_lenient"]) for r in rs),
         "points_good": sum(bool(r["right"]) for r in rs), "points_good_lenient": sum(bool(r["right_lenient"]) for r in rs),
         "points_silent": sum(not r["said"] for r in rs), "points_only_wrong": sum(bool(r["said"]) and not r["right"] for r in rs),
         "leader_said": len(led), "leader_right": sum(r["leader_right"] for r in led), "leader_right_lenient": sum(r["leader_right_lenient"] for r in led),
         "leader_rank": {str(k): sum(r["leader_rank"] == k for r in led) for k in (1, 2, 3)}}
    s["mean_lenient_size"] = round(sum(r["lenient_size"] for r in rs) / len(rs), 2) if rs else None
    s["mean_front_size"] = round(sum(r["front_size"] for r in rs) / len(rs), 2) if rs else None
    s.update(A=pct(s["right"], n_said), A_lenient=pct(s["right_lenient"], n_said), B=pct(s["points_good"], len(rs)), B_lenient=pct(s["points_good_lenient"], len(rs)),
             leader=pct(s["leader_right"], len(led)), leader_lenient=pct(s["leader_right_lenient"], len(led)), leader_coverage=pct(len(led), len(rs)))
    return s


def window(rs, name):
    lo, hi = fc.WINDOWS[name]
    return [r for r in rs if (lo <= r["tau"] <= hi if name != "far" else lo < r["tau"] <= hi)]


def interval(rs, key, n=2000, seed=20261118):
    races = sorted({(r["day"], r["race"]) for r in rs})
    by = {k: [r for r in rs if (r["day"], r["race"]) == k] for k in races}
    rng, vals = random.Random(seed), []
    for _ in range(n):
        sample = [x for k in rng.choices(races, k=len(races)) for x in by[k]]
        v = summarise(sample)[key]
        if v is not None:
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None


def main():
    out = Path(sys.argv[1])
    args = sys.argv[2:]
    configs, days = args[:args.index("--")], args[args.index("--") + 1:]
    res = {"days": days, "configs": {}}
    for c in configs:
        every = [r for d in days for r in rows(c, d)]
        allr = [r for r in every if r["clock_agrees"]]          # main analysis: races whose clock agrees with the official time
        res["configs"][c] = {"races": len({(r["day"], r["race"]) for r in allr}), "races_run": len({(r["day"], r["race"]) for r in every}),
                             "every_race_run": {w: {k: v for k, v in summarise(window(every, w)).items() if k in ("points", "said", "right", "A", "B", "A_lenient", "B_lenient", "leader_said", "leader_right", "leader")} for w in fc.WINDOWS}}
        for w in fc.WINDOWS:
            rs = window(allr, w)
            s = summarise(rs)
            s["interval"] = {k: interval(rs, k) for k in ("A", "B", "leader", "A_lenient", "B_lenient")}
            s["by_day"] = {d: {k: v for k, v in summarise([r for r in rs if r["day"] == d]).items() if k in ("points", "said", "right", "A", "B", "A_lenient", "B_lenient", "leader_said", "leader_right", "leader")} for d in days}
            res["configs"][c][w] = s
        res["configs"][c]["by_tau"] = {str(t): {k: v for k, v in summarise([r for r in allr if r["tau"] == t]).items() if k in ("points", "said", "right", "A", "B", "A_lenient", "B_lenient", "leader_said", "leader_right", "points_silent")} for t in range(36, -1, -3)}
        res["configs"][c]["winner_call"] = {}
        for t in (3, 0):                 # at the finish: is the official winner among the stated numbers, is the stated leader the winner
            rs = [r for r in allr if r["tau"] == t]
            led = [r for r in rs if r["leader"] is not None]
            res["configs"][c]["winner_call"][str(t)] = {"races": len(rs), "said_something": sum(bool(r["said"]) for r in rs), "winner_among_said": sum(r["winner"] in r["said"] for r in rs),
                                                        "leader_said": len(led), "leader_is_winner": sum(r["leader"] == r["winner"] for r in led)}
        print(f"\n== {c}  ({res['configs'][c]['races']} races)")
        for w in fc.WINDOWS:
            s = res["configs"][c][w]
            print(f"  {w:5s} points {s['points']:4d}  A {s['A']}% ({s['right']}/{s['said']}) {s['interval']['A']}  B {s['B']}% ({s['points_good']}/{s['points']}) {s['interval']['B']}"
                  f"  lenient A {s['A_lenient']}% B {s['B_lenient']}%  leader {s['leader']}% ({s['leader_right']}/{s['leader_said']}) lenient {s['leader_lenient']}%  silent {s['points_silent']}")
            print("        by day: " + "  ".join(f"{d}: A {v['A']} B {v['B']}" for d, v in s["by_day"].items()))
        print("  winner call: " + "  ".join(f"{t} s to go: winner among said {v['winner_among_said']}/{v['races']} (said something {v['said_something']}), leader is winner {v['leader_is_winner']}/{v['leader_said']}" for t, v in res["configs"][c]["winner_call"].items()))
        print("  mean size of the lenient first three: " + "  ".join(f"{w} {res['configs'][c][w]['mean_lenient_size']}" for w in fc.WINDOWS) + "; of the lenient front: " + "  ".join(f"{w} {res['configs'][c][w]['mean_front_size']}" for w in fc.WINDOWS))
        print("  every race run (" + str(res["configs"][c]["races_run"]) + "): " + "  ".join(f"{w}: A {v['A']} B {v['B']}" for w, v in res["configs"][c]["every_race_run"].items()))
        print("  by seconds to go (B, A): " + "  ".join(f"{t}:{v['B']}/{v['A']}" for t, v in res["configs"][c]["by_tau"].items()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
