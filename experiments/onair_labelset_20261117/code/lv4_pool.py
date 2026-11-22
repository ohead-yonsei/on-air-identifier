"""live3: compare the live runner without (A) and with (B) the finish-stretch leader model on unused days; verdicts of live3/DESIGN_KO.md.
usage: lv4_pool.py LIVE3_DIR DAY [DAY ...]      reads LIVE3_DIR/{A,B}/DAY/RNN/{live.json,score.json,official.json}; standard library only
A race enters when its clock agrees with the official winning time (fn_common.video_finish rule) in both arms."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn_common as fc

D, days = Path(sys.argv[1]), sys.argv[2:]
KEYS = ["lines", "numbers", "numbers right", "numbers close", "numbers wrong", "lines with a wrong number", "leader lines", "leader right", "leader close", "leader wrong"]


def race(arm, day, r):
    p = D / arm / day / r
    if not (p / "live.json").exists() or not (p / "score.json").exists():
        return {"status": "no run"}
    live, off, sc = (json.loads((p / f).read_text()) for f in ("live.json", "official.json", "score.json"))
    c, tw = live["clock"], fc.winner_time(off)
    if c["start"] is None or c["finish"] is None or tw is None:
        return {"status": "clock not found"}
    agree = abs(c["finish"] + 0.2 - c["start"] - tw - 0.15) <= 0.35
    pts = [e for e in live["events"] if e["kind"] == "point"]
    fin = next((e for e in live["events"] if e["kind"] == "finish"), None)
    stretch = {"leader lines": 0, "leader right": 0, "leader close": 0, "leader wrong": 0}      # leader lines within 36 s of the clock stop
    for row in sc["rows"]:
        if "verdict" not in row and row.get("leader") and c["finish"] - row["video_time"] <= 36.0:
            stretch["leader lines"] += 1
            stretch["leader " + row["leader_verdict"]] += 1
    win = next((x for x in sc["rows"] if "verdict" in x), None)
    return {"status": "clock agrees" if agree else "clock disagrees", "totals": {k: sc["totals"].get(k, 0) for k in KEYS}, "stretch": stretch,
            "delays": [e["delay"] for e in pts], "finish_after": fin["seconds_after_the_finish"] if fin else None,
            "winner": win["verdict"] if win else "no finish line"}


rows = []
for day in days:
    for p in sorted((D / "A" / day).glob("R*")):
        a, b = race("A", day, p.name), race("B", day, p.name)
        rows.append({"day": day, "race": p.name, "A": a, "B": b, "in": a["status"] == "clock agrees" and b["status"] == "clock agrees"})
inc = [x for x in rows if x["in"]]


def pool(arm, xs):
    t = {k: sum(x[arm]["totals"][k] for x in xs) for k in KEYS}
    s = {k: sum(x[arm]["stretch"][k] for x in xs) for k in ("leader lines", "leader right", "leader close", "leader wrong")}
    d = sorted(v for x in xs for v in x[arm]["delays"])
    sh = lambda a, b: round(a / b, 4) if b else None
    return {"totals": t, "stretch_last_36s": s, "numbers_right": sh(t["numbers right"], t["numbers"]), "leader_right": sh(t["leader right"], t["leader lines"]),
            "leader_wrong": sh(t["leader wrong"], t["leader lines"]), "stretch_leader_right": sh(s["leader right"], s["leader lines"]),
            "delay_median": d[len(d) // 2] if d else None, "delay_max_per_race_ok": all(x[arm]["delays"] and max(x[arm]["delays"]) <= 2.0 for x in xs),
            "finish_line_ok": all(x[arm]["finish_after"] is not None and x[arm]["finish_after"] <= 3.0 for x in xs),
            "winner": {k: sum(x[arm]["winner"] == k for x in xs) for k in ("right", "wrong", "no call")}}


A, B = pool("A", inc), pool("B", inc)
diff = None if A["leader_right"] is None or B["leader_right"] is None else round(B["leader_right"] - A["leader_right"], 4)
rng, boot = random.Random(20261123), []
for _ in range(2000 if inc else 0):                              # races resampled with replacement (descriptive only)
    xs = [rng.choice(inc) for _ in inc]
    a, b = pool("A", xs), pool("B", xs)
    if a["leader_right"] is not None and b["leader_right"] is not None:
        boot.append(b["leader_right"] - a["leader_right"])
boot.sort()
ci = [round(boot[int(0.025 * len(boot))], 4), round(boot[int(0.975 * len(boot)) - 1], 4)] if boot else None
v = {
    "V1 B leader right >= A + 5 pt and fewer wrong": diff is not None and diff >= 0.05 and B["totals"]["leader wrong"] < A["totals"]["leader wrong"],
    "V2 B leader right >= 75%": B["leader_right"] is not None and B["leader_right"] >= 0.75,
    "V3 B numbers >= 90% and speed": B["numbers_right"] is not None and B["numbers_right"] >= 0.90 and B["delay_median"] is not None
                                     and B["delay_median"] <= 0.5 and B["delay_max_per_race_ok"] and B["finish_line_ok"],
    "V4 A numbers >= 90% and speed (live2 replication)": A["numbers_right"] is not None and A["numbers_right"] >= 0.90 and A["delay_median"] is not None
                                     and A["delay_median"] <= 0.5 and A["delay_max_per_race_ok"] and A["finish_line_ok"],
}
res = {"days": days, "races": [{k: x[k] for k in ("day", "race", "in")} | {"A_status": x["A"]["status"], "B_status": x["B"]["status"]} for x in rows],
       "races_in": len(inc), "A": A, "B": B, "B_minus_A_leader_right": diff, "bootstrap_95": ci, "verdicts": v,
       "recommend B as live_run default": v["V1 B leader right >= A + 5 pt and fewer wrong"] and v["V3 B numbers >= 90% and speed"]}
name = "pooled_" + "_".join(days) + ".json"
(D / name).write_text(json.dumps(res, ensure_ascii=False, indent=1))
for x in rows:
    print(x["day"], x["race"], "in" if x["in"] else "out", x["A"]["status"], {arm: (x[arm].get("totals", {}).get("leader right"), x[arm].get("totals", {}).get("leader lines"),
          x[arm].get("totals", {}).get("leader wrong")) for arm in "AB"})
print(json.dumps({k: res[k] for k in res if k != "races"}, ensure_ascii=False, indent=1))
