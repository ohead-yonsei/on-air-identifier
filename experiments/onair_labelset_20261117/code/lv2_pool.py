"""live2: pool the per-race scores of a live run on an unused day and apply the verdicts locked in live2/DESIGN_KO.md.
usage: lv2_pool.py LIVE2_DIR       (LIVE2_DIR/RNN/{live.json,score.json,official.json}); standard library only
Only races whose clock agrees with the official winning time (rule of fn_common.video_finish) enter the verdicts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn_common as fc

D = Path(sys.argv[1])
races, pooled, delays = [], None, []
for r in sorted(p for p in D.iterdir() if p.is_dir() and p.name.startswith("R")):
    row = {"race": r.name}
    if not (r / "live.json").exists():
        row["status"] = "no run (see run.log)"
        races.append(row)
        continue
    live, off = json.loads((r / "live.json").read_text()), json.loads((r / "official.json").read_text())
    c, tw = live["clock"], fc.winner_time(off)
    if c["start"] is None or c["finish"] is None or tw is None:
        row["status"] = "clock not found"
        races.append(row)
        continue
    run = c["finish"] + 0.2 - c["start"]
    agree = abs(run - tw - 0.15) <= 0.35
    sc = json.loads((r / "score.json").read_text())
    pts = [e["delay"] for e in live["events"] if e["kind"] == "point"]
    fin = next((e for e in live["events"] if e["kind"] == "finish"), None)
    win = next((x for x in sc["rows"] if "verdict" in x), None)
    row.update({"status": "clock agrees" if agree else "clock disagrees", "clock_run_minus_official": round(run - tw - 0.15, 2),
                "totals": sc["totals"], "delay_median": sorted(pts)[len(pts) // 2] if pts else None, "delay_max": max(pts) if pts else None,
                "finish_line_after_clock_stop": fin["seconds_after_the_finish"] if fin else None,
                "furthest_behind": live["furthest_behind_seconds"], "dense": live["dense_pictures"],
                "winner_call": win["winner_call"] if win else None, "official_winner": sc["official_winner"], "winner_verdict": win["verdict"] if win else "no finish line"})
    races.append(row)
    if agree:
        pooled = {k: 0 for k in sc["totals"]} if pooled is None else pooled
        for k, v in sc["totals"].items():
            pooled[k] += v
        delays += pts

ok = [x for x in races if x["status"] == "clock agrees"]
med = sorted(delays)[len(delays) // 2] if delays else None
calls = [x["winner_verdict"] for x in ok]
p = pooled or {}
share = lambda a, b: round(a / b, 4) if b else None
v = {
    "V1 speed": bool(ok) and med is not None and med <= 0.5 and all(x["delay_max"] is not None and x["delay_max"] <= 2.0 for x in ok)
                and all(x["finish_line_after_clock_stop"] is not None and x["finish_line_after_clock_stop"] <= 3.0 for x in ok),
    "V2 numbers right >= 90%": bool(p.get("numbers")) and p["numbers right"] / p["numbers"] >= 0.90,
    "V3 leader right >= 75%": bool(p.get("leader lines")) and p["leader right"] / p["leader lines"] >= 0.75,
    "V4 wrong winner calls <= 1": calls.count("wrong") <= 1,
}
res = {"races": races, "clock_agreeing_races": len(ok), "pooled": p, "pooled_point_delay_median": med,
       "numbers_right_share": share(p.get("numbers right", 0), p.get("numbers", 0)),
       "numbers_right_or_close_share": share(p.get("numbers right", 0) + p.get("numbers close", 0), p.get("numbers", 0)),
       "leader_right_share": share(p.get("leader right", 0), p.get("leader lines", 0)),
       "leader_right_or_close_share": share(p.get("leader right", 0) + p.get("leader close", 0), p.get("leader lines", 0)),
       "winner_calls": {"right": calls.count("right"), "wrong": calls.count("wrong"), "no call": calls.count("no call")},
       "verdicts": v, "V1-V3 all met": v["V1 speed"] and v["V2 numbers right >= 90%"] and v["V3 leader right >= 75%"]}
(D / "pooled.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
for x in races:
    print(x["race"], x["status"], {k: x.get(k) for k in ("clock_run_minus_official", "delay_median", "delay_max", "finish_line_after_clock_stop", "winner_call", "official_winner", "winner_verdict")},
          x.get("totals"))
print(json.dumps({k: res[k] for k in res if k != "races"}, ensure_ascii=False, indent=1))
