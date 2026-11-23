"""v12 test: score the live runs with --v12 against the official passing times, and apply the verdicts of v12/DESIGN_KO.md.
usage: v12_score.py DIR DAY [DAY ...]      reads DIR/DAY/RNN/{live.json,official.json}; standard library only

Every time point of a run carries both the v11 statements (what v11's lines would say: the numbers in the leading group and the leader)
and v12's first five as said (said_order: groups front to back; horses in one group are "side by side"). Both are scored on the same
points against the order computed from the official passing times at that moment (race time = official winning time minus the time
left to the clock stop, as code/lv1_score.py). Only races whose clock agrees with the official winning time are in the verdicts
(the rule of live2-live4)."""
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn_common as fc
from v12_train import truth

MARGIN = fc.MARGIN_M
D, days = Path(sys.argv[1]), sys.argv[2:]


def card_number(card, n):
    """A cap of 11 stands for 11 and up: said only when the race card has exactly one such runner (as live_run.Lines.number)."""
    if n < 11:
        return n
    high = [r["number"] for r in card["runners"] if r["number"] >= 11]
    return high[0] if len(high) == 1 else None


def names_score(named, tr):
    s = Counter(points=1, named=len(named))
    for n in named:
        rk, b = tr["rank"](n), tr["behind"](n)
        s["right"] += int(rk is not None and rk <= 5)
        s["right_lenient"] += int(b is not None and b <= tr["fifth"] + MARGIN)
    s["wrong_point"] += int(s["right"] < len(named))
    s["wrong_point_lenient"] += int(s["right_lenient"] < len(named))
    return s


def pairs_score(groups, tr):
    s = Counter()
    flat = [(gi, n) for gi, g in enumerate(groups) for n in g]
    for (ga, a), (gb, b) in itertools.combinations(flat, 2):
        ba, bb = tr["behind"](a), tr["behind"](b)
        if ba is None or bb is None:
            continue
        if ga == gb:
            s["side_by_side"] += 1
            s["side_by_side_within_3m"] += int(abs(ba - bb) <= MARGIN)
        else:
            s["pairs"] += 1
            s["pairs_right"] += int(ba < bb)
            s["pairs_right_lenient"] += int(ba <= bb + MARGIN)
    return s


def race(day, r):
    p = D / day / r
    if not (p / "live.json").exists():
        return {"status": "no run"}
    live, off = json.loads((p / "live.json").read_text()), json.loads((p / "official.json").read_text())
    c, tw, card = live["clock"], fc.winner_time(off), live["racecard"]
    if c["start"] is None or c["finish"] is None or tw is None:
        return {"status": "clock not found"}
    agree = abs(c["finish"] + 0.2 - c["start"] - tw - 0.15) <= 0.35
    v11, v12, parts = Counter(), Counter(), {}
    for e in live["events"]:
        if e["kind"] != "point":
            continue
        rt = tw - (c["finish"] - e["video_time"])
        tr = truth(off, rt) if rt >= 0 else None
        if tr is None:
            continue
        n11 = {card_number(card, n) for n in e["statements"]} | ({card_number(card, e["leader"])} if e.get("leader") else set())
        s11, s12 = names_score(sorted(x for x in n11 if x), tr), names_score([n for g in e["said_order"] for n in g], tr)
        s12.update(pairs_score(e["said_order"], tr))
        v11.update(s11)
        v12.update(s12)
        prt = "start" if rt < 18 else "finish" if c["finish"] - e["video_time"] <= 36 else "middle"
        parts.setdefault(prt, [Counter(), Counter()])
        parts[prt][0].update(s11)
        parts[prt][1].update(s12)
    pts = [e for e in live["events"] if e["kind"] == "point"]
    fin = next((e for e in live["events"] if e["kind"] == "finish"), None)
    return {"status": "clock agrees" if agree else "clock disagrees", "v11": v11, "v12": v12, "parts": parts, "delays": [e["delay"] for e in pts],
            "finish_after": fin["seconds_after_the_finish"] if fin else None}


def summary(c):
    p = c["points"] or 1
    pc = lambda a, b: round(100 * a / b, 2) if b else None
    return {"points": c["points"], "named_per_point": round(c["named"] / p, 3), "right_per_point": round(c["right"] / p, 3),
            "named_right_%": pc(c["right"], c["named"]), "wrong_point_%": pc(c["wrong_point"], c["points"]),
            "wrong_point_lenient_%": pc(c["wrong_point_lenient"], c["points"]), "pairs_per_point": round(c["pairs"] / p, 3),
            "pairs_right_%": pc(c["pairs_right"], c["pairs"]), "pairs_right_lenient_%": pc(c["pairs_right_lenient"], c["pairs"]),
            "side_by_side_per_point": round(c["side_by_side"] / p, 3), "side_by_side_within_3m_%": pc(c["side_by_side_within_3m"], c["side_by_side"])}


rows = []
for day in days:
    for p in sorted((D / day).glob("R*")):
        x = race(day, p.name)
        rows.append({"day": day, "race": p.name, **x})
inc = [x for x in rows if x["status"] == "clock agrees"]
tot = lambda key, xs: sum((x[key] for x in xs), Counter())
A, B = summary(tot("v11", inc)), summary(tot("v12", inc))
by_part = {prt: {"v11": summary(sum((x["parts"][prt][0] for x in inc if prt in x["parts"]), Counter())),
                 "v12": summary(sum((x["parts"][prt][1] for x in inc if prt in x["parts"]), Counter()))} for prt in ("start", "middle", "finish")}
delays = sorted(v for x in inc for v in x["delays"])
speed = {"delay_median": delays[len(delays) // 2] if delays else None, "delay_max_per_race_ok": all(x["delays"] and max(x["delays"]) <= 2.0 for x in inc),
         "finish_line_ok": all(x["finish_after"] is not None and x["finish_after"] <= 3.0 for x in inc)}
rng, boot = random.Random(20261124), []
for _ in range(2000 if inc else 0):                      # races resampled with replacement (descriptive only)
    xs = [rng.choice(inc) for _ in inc]
    a, b = tot("v11", xs), tot("v12", xs)
    if a["points"]:
        boot.append(b["right"] / b["points"] - a["right"] / a["points"])
boot.sort()
ci = [round(boot[int(0.025 * len(boot))], 3), round(boot[int(0.975 * len(boot)) - 1], 3)] if boot else None
v = {"V1 v12 points with a wrong number (first five, strict) <= 5.0%": B["wrong_point_%"] is not None and B["wrong_point_%"] <= 5.0,
     "V2 v12 right numbers per point >= v11 + 0.3 and bootstrap lower end > 0": B["right_per_point"] >= A["right_per_point"] + 0.3 and ci is not None and ci[0] > 0,
     "V3 v12 pair orders right (strict) >= 85%": B["pairs_right_%"] is not None and B["pairs_right_%"] >= 85.0,
     "V4 speed (median <= 0.5 s, max per race <= 2.0 s, finish line <= 3.0 s)": speed["delay_median"] is not None and speed["delay_median"] <= 0.5
                                                                            and speed["delay_max_per_race_ok"] and speed["finish_line_ok"]}
res = {"days": days, "races": [{k: x[k] for k in ("day", "race", "status")} for x in rows], "races_in": len(inc), "v11": A, "v12": B, "by_part": by_part,
       "speed": speed, "v12_minus_v11_right_per_point_bootstrap_95": ci, "verdicts": v, "recommend v12 as live_run default": all(v.values())}
(D / ("pooled_" + "_".join(days) + ".json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps({k: res[k] for k in res if k != "races"}, ensure_ascii=False, indent=1))
for x in rows:
    print(x["day"], x["race"], x["status"], summary(x["v11"]) if "v11" in x else "", summary(x["v12"]) if "v12" in x else "")
