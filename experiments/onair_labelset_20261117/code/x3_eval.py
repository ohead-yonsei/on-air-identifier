"""far: the retrained cap classifier against v11's, on the same riders (same boxes).
usage: x3_eval.py OUT.json NAME DAY [DAY ...]        NAME = folder under far/dense (x2_reprob.py); compared with A = v11's classifier
  winner call      the rule of the winner experiment (w4_winner_guard.py) on the pictures of the last seconds, every race run
  whole race       for the days that have whole-race pictures: what h5_eval_caps.py measures (true leader readable, leader right / wrong,
                   numbers A / B), with "current" = v11's classifier"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import h5_eval_caps as h5
import w4_winner_guard as w4

out, name, days = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]


def last(day, which):
    if which == "A":
        return w4.dense(day)
    return json.loads((ROOT / "far" / "dense" / which / "last" / f"{day}.json").read_text())["races"]


res = {"days": days, "new": name, "winner": {}, "rows": []}
tot = {"A": Counter(), name: Counter()}
per_day = {}
for d in days:
    fc.FOLDER.setdefault(d, d)
    off = fc.official(d)
    a, b = last(d, "A"), last(d, name)
    for race in sorted(a):
        rank = {h["number"]: h["rank"] for h in off[race]["horses"] if h.get("rank")}
        win = next((x for x, r in rank.items() if r == 1), None)
        row = {"day": d, "race": race, "winner": win, "clock_agrees": a[race]["clock_agrees"]}
        for which, src in (("A", a), (name, b)):
            n, info = w4.call(src[race]["pictures"])
            kind = "none" if n is None else ("right" if n == win else "wrong")
            tot[which][kind] += 1
            per_day.setdefault(d, {}).setdefault(which, Counter())[kind] += 1
            row[which] = {"call": n, "candidate": info["candidate"], "kind": kind}
        res["rows"].append(row)
        if row["A"]["call"] != row[name]["call"]:
            print("  ", d, race, "winner", win, "| A", row["A"]["call"], f"(candidate {row['A']['candidate']})", "|", name, row[name]["call"], f"(candidate {row[name]['candidate']})")
for which, c in tot.items():
    res["winner"][which] = dict(c)
    print(f"winner call, {which:3s}: right {c['right']} wrong {c['wrong']} none {c['none']} of {sum(c.values())}")
print("  per day (right+wrong):", {d[-2:]: {w: f"{c['right']}+{c['wrong']}w" for w, c in v.items()} for d, v in per_day.items()})
whole = [d for d in days if (ROOT / "far" / "dense" / name / "whole" / f"{d}.json").exists()]
if whole:
    h5.path = lambda nm, kind, day: (ROOT / "caps2" / "dense" / "A" / kind / f"{day}.json") if nm == "current" else (ROOT / "far" / "dense" / nm / kind / f"{day}.json")
    print("whole race on", whole, "(current = v11's classifier)")
    sys.argv = [sys.argv[0], str(out.with_name(out.stem + "_whole.json")), name] + whole
    h5.main()
out.write_text(json.dumps(res, indent=1))
