"""2026-11-15: the program's output as L1 (results_order), F1 (results_fast) and F2 (results_f2) against the strip. Standard library only.
usage: nc5_score_day_programs.py DAY      Additional description, not a locked test: L1's statements on this day were scored before.
The strip is taken from results_order (the only run that read it), joined by time point; only time points with the strip on screen count."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "live_commentary" / sys.argv[1]
out = {}
for name in ("results_order", "results_fast", "results_f2"):
    if not (O / name).exists():
        continue
    c = {"races": 0, "strip_points": 0, "numbers": 0, "numbers_right": 0, "points_good": 0, "points_silent": 0, "leader_said": 0, "leader_right": 0,
         "pairs_said": 0, "pairs_right": 0, "fallback_numbers": 0, "fallback_right": 0, "seconds": 0.0, "time_points": 0}
    for f in sorted((O / name).glob("R??.json")):
        d = json.loads(f.read_text())
        ref = {round(p["t"], 2): p.get("strip") for p in json.loads((O / "results_order" / f.name).read_text())["points"]}
        c["races"] += 1
        c["seconds"] += d.get("seconds", 0)
        c["time_points"] += len(d["points"])
        for p in d["points"]:
            S = ref.get(round(p["t"], 2))
            if not S:
                continue
            pos = {n: i for i, n in enumerate(S)}
            r = sum(n in S[:3] for n in p["statements"])
            c["strip_points"] += 1
            c["numbers"] += len(p["statements"])
            c["numbers_right"] += r
            c["points_good"] += r > 0
            c["points_silent"] += not p["statements"]
            for x in p["detail"]:
                if x["source"] == "fallback":
                    c["fallback_numbers"] += 1
                    c["fallback_right"] += x["number"] in S[:3]
            if p.get("leader") is not None:
                c["leader_said"] += 1
                c["leader_right"] += p["leader"] == S[0]
            for o in p.get("order", []):
                if o["ahead"] in pos or o["behind"] in pos:
                    c["pairs_said"] += 1
                    c["pairs_right"] += pos.get(o["ahead"], 99) < pos.get(o["behind"], 99)
    c["seconds_per_time_point"] = round(c["seconds"] / c["time_points"], 2) if c["time_points"] else None
    out[name] = c
    print(f"{name:14s} races {c['races']} strip points {c['strip_points']}  A {c['numbers_right']}/{c['numbers']}  B {c['points_good']}/{c['strip_points']}  silent {c['points_silent']}  "
          f"leader {c['leader_right']}/{c['leader_said']}  pairs {c['pairs_right']}/{c['pairs_said']}  fallback {c['fallback_right']}/{c['fallback_numbers']}  "
          f"{c['seconds']:.0f} s for {c['time_points']} points = {c['seconds_per_time_point']} s per point")
(O / "score_programs.json").write_text(json.dumps(out, indent=1))
