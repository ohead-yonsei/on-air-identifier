"""Script demo, step 4: check the script against the official passing times (done after the script exists; the script never saw them).
usage: python3 code/ls4_score.py DEMO_DIR        (standard library only)

Per line: every "N번" in it is compared with the first three at that race time (fn_common.reference_full); the leader the line
was allowed to claim is compared with the horse in front; the winner call with the official winner.
right = in the computed first three (leader: is first); close = within 3 m of the third (of the first); wrong = neither.
Writes score.json."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn_common as fc

D = fc.ROOT / sys.argv[1]
off = json.loads((D / "official.json").read_text())
script = json.loads((D / "script.json").read_text())
ident = json.loads((D / "ident.json").read_text())
tw = fc.winner_time(off)
winner = next(h["number"] for h in off["horses"] if h["rank"] == 1)
rows, tot = [], {"lines": 0, "ranking_lines": 0, "numbers": 0, "numbers_right": 0, "numbers_close": 0, "numbers_wrong": 0, "lines_with_wrong_number": 0,
                 "leader_claims": 0, "leader_right": 0, "leader_close": 0, "leader_wrong": 0}
for r in script["rows"]:
    rt = r["elapsed"]
    ref = fc.reference_full(off, rt) if rt <= tw + 0.3 else None
    nums = sorted({int(n) for n in re.findall(r"(\d+)\s*번", r["line"])})
    out = {"k": r["k"], "elapsed": rt, "numbers": {}, "leader": None, "top3": ref["top3"] if ref else None, "order": ref["order"] if ref else None}
    tot["lines"] += 1
    if r["mode"] == "finish":
        call = r["given"]["winner"]
        out["winner_call"] = {"number": call, "official_winner": winner, "verdict": "no call" if call is None else ("right" if call == winner else "wrong")}
    elif ref:
        tot["ranking_lines"] += r["mode"] == "ranking"
        recall = set(r["given"].get("recall") or [])
        for n in nums:
            if n in recall and r["mode"] == "confirmed":
                out["numbers"][n] = "recalled"            # "a moment ago": not a claim about now
                continue
            v = "right" if n in ref["top3"] else "close" if n in ref["top3_lenient"] else "wrong"
            out["numbers"][n] = v
            tot["numbers"] += 1
            tot["numbers_" + v] += 1
        tot["lines_with_wrong_number"] += any(v == "wrong" for v in out["numbers"].values())
        lead = r["given"]["leader"]
        if lead is not None:
            v = "right" if lead == ref["leader"] else "close" if lead in ref["leader_lenient"] else "wrong"
            out["leader"] = {"number": lead, "verdict": v, "official_leader": ref["leader"], "rank_then": ref["order"].index(lead) + 1 if lead in ref["order"] else None}
            tot["leader_claims"] += 1
            tot["leader_" + v] += 1
    rows.append(out)
clock = ident["clock"]
res = {"winner_time": tw, "clock_ran": round(clock["stop"] - clock["start"], 1), "official_first_three": [h["number"] for h in off["horses"] if h["rank"] in (1, 2, 3)],
       "winner_call": next((x["winner_call"] for x in rows if "winner_call" in x), None), "totals": tot, "rows": rows}
(D / "score.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
