"""AI commentary demo, step 3: count the horse numbers named in the commentary against the strip. See live_commentary/DESIGN_KO.md.
usage: cl3_score.py DAY      Standard library only.

Scored only where the strip was on screen at that time point (the commentary system itself never saw the strip):
  right = in the strip's first three, places_4_5 = fourth or fifth on the strip, off_strip = not on the strip.
System = numbers in leader-mode sentences. Picture only = numbers in the comparison sentences (every time point)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "live_commentary" / sys.argv[1]
SUB = sys.argv[2] if len(sys.argv) > 2 else ""      # "_v2" for the second run


def blank():
    return {"points": 0, "points_with_strip": 0, "numbers": 0, "right": 0, "places_4_5": 0, "off_strip": 0, "no_reference": 0, "points_with_right": 0, "points_with_wrong": 0}


def add(c, nums, strip):
    c["points"] += 1
    if not strip:
        c["no_reference"] += len(nums)
        return
    c["points_with_strip"] += 1
    r = [n for n in nums if n in strip[:3]]
    c["numbers"] += len(nums)
    c["right"] += len(r)
    c["places_4_5"] += sum(n in strip[3:] for n in nums)
    c["off_strip"] += sum(n not in strip for n in nums)
    c["points_with_right"] += bool(r)
    c["points_with_wrong"] += len(r) < len(nums)


total = {"system": blank(), "picture_only": blank()}
modes = {"leader": 0, "confirmed": 0}
sources = {"model": 0, "model_retry": 0, "template": 0}
recalls = 0
races = {}
for f in sorted((O / f"commentary{SUB}").glob("R*.json")):
    d = json.loads(f.read_text())
    per = {"system": blank(), "picture_only": blank(), "info": d["info"], "points": len(d["rows"]), "leader_points": sum(r["mode"] == "leader" for r in d["rows"])}
    for r in d["rows"]:
        modes[r["mode"]] += 1
        sources[r["source"]] += 1
        recalls += bool(r["mode"] == "confirmed" and r["said_numbers"])
        for key, nums in (("system", r["said_numbers"] if r["mode"] == "leader" else []), ("picture_only", r["picture_only_numbers"])):
            add(per[key], nums, r["strip"])
            add(total[key], nums, r["strip"])
    races[d["race"]] = per
for c in total.values():
    c["share_right"] = c["right"] / c["numbers"] if c["numbers"] else None
    c["share_points_with_right"] = c["points_with_right"] / c["points_with_strip"] if c["points_with_strip"] else None
out = {"day": sys.argv[1], "races": len(races), "time_points": sum(modes.values()), "modes": modes, "sentence_sources": sources, "confirmed_sentences_recalling_numbers": recalls,
       "total": total, "by_race": races}
(O / f"score{SUB}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("races", len(races), "time points", out["time_points"], "modes", modes, "sources", sources, "recalls", recalls)
for k, c in total.items():
    print(f"  {k:12s}: scored points {c['points_with_strip']}, numbers {c['numbers']}, right {c['right']}, places 4-5 {c['places_4_5']}, off strip {c['off_strip']}, "
          f"no reference {c['no_reference']}, points with a right number {c['points_with_right']}, with a wrong number {c['points_with_wrong']}")
