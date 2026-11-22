"""live1: check the lines of a live run against the official passing times (after the run; the run never saw them).
usage: lv1_score.py LIVE_DIR OFFICIAL.json       standard library only
A time point's race time is the official winning time minus the time left to the clock stop (as the benchmark's official track).
right = in the computed first three (leader: is first); close = within 3 m of the third (of the first); wrong = neither."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn_common as fc

D, off = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
live = json.loads((D / "live.json").read_text())
tw, finish = fc.winner_time(off), live["clock"]["finish"]
winner = next(h["number"] for h in off["horses"] if h["rank"] == 1)
tot = {"lines": 0, "lines with a number": 0, "numbers": 0, "numbers right": 0, "numbers close": 0, "numbers wrong": 0, "lines with a wrong number": 0,
       "leader lines": 0, "leader right": 0, "leader close": 0, "leader wrong": 0}
rows = []
for e in live["events"]:
    tot["lines"] += 1
    if e["kind"] == "finish":
        rows.append({"video_time": e["video_time"], "line": e["line"], "winner_call": e["winner"], "official_winner": winner,
                     "verdict": "no call" if e["winner"] is None else "right" if e["winner"] == winner else "wrong"})
        continue
    if e["kind"] != "point":
        continue
    ref = fc.reference_full(off, tw - (finish - e["video_time"]))
    nums = sorted(set(e["statements"]) | ({e["leader"]} if e.get("leader") else set()))
    v = {n: "right" if n in ref["top3"] else "close" if n in ref["top3_lenient"] else "wrong" for n in nums}
    lead = None if not e.get("leader") else "right" if e["leader"] == ref["leader"] else "close" if e["leader"] in ref["leader_lenient"] else "wrong"
    tot["lines with a number"] += bool(nums)
    tot["numbers"] += len(nums)
    for k in ("right", "close", "wrong"):
        tot[f"numbers {k}"] += sum(x == k for x in v.values())
    tot["lines with a wrong number"] += any(x == "wrong" for x in v.values())
    if lead:
        tot["leader lines"] += 1
        tot[f"leader {lead}"] += 1
    rows.append({"video_time": e["video_time"], "line": e["line"], "numbers": v, "leader": e.get("leader"), "leader_verdict": lead, "official_order": ref["order"][:4]})
res = {"official_winner": winner, "totals": tot, "delay_seconds": live["delay_seconds"], "rows": rows}
(D / "score.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(tot)
for r in rows:
    if "verdict" in r:
        print("finish:", r["winner_call"], "official", r["official_winner"], r["verdict"])
    elif any(x != "right" for x in r["numbers"].values()) or r["leader_verdict"] in ("close", "wrong"):
        print(f"  {r['video_time']:6.2f} numbers {r['numbers']} leader {r['leader']} {r['leader_verdict']} | official first four {r['official_order']} | {r['line']}")
