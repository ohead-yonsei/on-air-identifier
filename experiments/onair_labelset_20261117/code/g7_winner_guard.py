"""finish3 stage 2 (exploration on days seen before): can a stricter rule keep wrong winner calls out?
usage: g7_winner_guard.py OUT.json            (CPU only; reads finish2/dense/DAY.json, the pictures of the last 37 s, 5 a second)

Stage 1 of finish3 found wrong calls on the test days (a horse that finished ninth, named in 5-6 of the 31 pictures). The test days are not
used here. On the development days (leave-day-out finish-stretch model) and the confirmation days of finish2, for the pictures of the last
6 s, every picture names the strip-trained model's leader pick (rule P:strip:6.0:5:1.0 = the base). Guards tried:
  need N        the most named number must be named in at least N pictures (base: 5)
  ratio R       and at least R times as often as the runner-up
  agree         and the finish-stretch model's most named number over the same pictures must be the same number
  late K        and the number must have been picked (by either model) in at least K pictures of the 6 s before those (12 s to 6 s to go)
All races run are counted (the clock-agrees subset is also given)."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn9_winner as f9

DEV = ["otherday", "otherday2", "otherday3", "otherday4", "otherday5", "20261115"]
CONF = ["otherday9", "otherday10", "otherday11"]


def races(day):
    off, a, b, out = fc.official(day), f9.picks(day, "P", "strip"), f9.picks(day, "P", "fs"), []
    for race in sorted(a):
        rank = {h["number"]: h["rank"] for h in off[race]["horses"]}

        def count(pk, lo, hi):
            return Counter(n for t, n, _ in pk if n is not None and lo - 0.05 <= t <= hi + 0.05)

        out.append({"day": day, "race": race, "agree": a[race]["agree"], "rank": rank,
                    "strip": count(a[race]["picks"], 0, 6), "fs": count(b[race]["picks"], 0, 6),
                    "before": count(a[race]["picks"], 6.2, 12) + count(b[race]["picks"], 6.2, 12)})
    return out


def call(r, need=5, ratio=1.0, agree=False, late=0):
    c = r["strip"].most_common(2)
    if not c or c[0][1] < need or (len(c) > 1 and c[0][1] < ratio * c[1][1]):
        return None
    n = c[0][0]
    if agree and (not r["fs"] or r["fs"].most_common(1)[0][0] != n):
        return None
    if late and r["before"].get(n, 0) < late:
        return None
    return n


def tally(rs, **kw):
    calls = [(call(r, **kw), r) for r in rs]
    called = [(n, r) for n, r in calls if n is not None]
    return {"races": len(rs), "called": len(called), "right": sum(r["rank"].get(n) == 1 for n, r in called),
            "wrong_ranks": sorted(r["rank"].get(n) or 99 for n, r in called if r["rank"].get(n) != 1)}


def main():
    data = {"dev": [r for d in DEV for r in races(d)], "conf": [r for d in CONF for r in races(d)]}
    rules = [("base need 5", {})] + [(f"need {n}", {"need": n}) for n in (6, 7, 8, 10)] + [(f"ratio {x}", {"ratio": x}) for x in (2.0, 3.0)] \
        + [("agree", {"agree": True}), ("agree + need 6", {"agree": True, "need": 6}), ("agree + ratio 2", {"agree": True, "ratio": 2.0})] \
        + [(f"late {k}", {"late": k}) for k in (1, 3, 5)] + [("agree + late 3", {"agree": True, "late": 3}), ("need 7 + ratio 2", {"need": 7, "ratio": 2.0})]
    res = {}
    for name, kw in rules:
        res[name] = {}
        line = f"{name:18s}"
        for key, rs in data.items():
            for sub, sel in (("all", rs), ("clock agrees", [r for r in rs if r["agree"]])):
                t = tally(sel, **kw)
                res[name][f"{key} {sub}"] = t
                line += f" | {key} {sub}: {t['called']}/{t['races']} called, {t['right']} right, wrong ranks {t['wrong_ranks']}"
        print(line)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
