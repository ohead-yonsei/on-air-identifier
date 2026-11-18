"""Finish stretch 2, stage 2: paired comparison of two variants on the same races (leader statements and A / B), last 36 s.
usage: fn12_compare.py OUT.json BASE NEW DAY [DAY ...]

Difference NEW - BASE of the strict leader accuracy, A and B, with a bootstrap over races (2000 resamples, seed 20261118).
Main analysis races only (clock agrees with the official time)."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn4_score as f4

out, base, new, days = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
R = {c: [r for d in days for r in f4.rows(c, d) if r["clock_agrees"]] for c in (base, new)}
races = sorted({(r["day"], r["race"]) for r in R[base]})
by = {c: {k: [r for r in R[c] if (r["day"], r["race"]) == k] for k in races} for c in (base, new)}
keys = ("leader", "leader_lenient", "A", "B")
point = {c: f4.summarise(R[c]) for c in (base, new)}
rng, diffs = random.Random(20261118), {k: [] for k in keys}
for _ in range(2000):
    pick = rng.choices(races, k=len(races))
    s = {c: f4.summarise([x for k in pick for x in by[c][k]]) for c in (base, new)}
    for k in keys:
        if s[base][k] is not None and s[new][k] is not None:
            diffs[k].append(s[new][k] - s[base][k])
res = {"days": days, "races": len(races), "base": base, "new": new, "point": {c: {k: point[c][k] for k in keys + ("leader_said", "leader_right", "said", "right", "points", "points_good")} for c in (base, new)}, "difference": {}}
for k in keys:
    v = sorted(diffs[k])
    res["difference"][k] = {"value": round(point[new][k] - point[base][k], 1), "interval": [round(v[int(0.025 * len(v))], 1), round(v[int(0.975 * len(v)) - 1], 1)]}
    print(f"{k:15s} {base} {point[base][k]}%  {new} {point[new][k]}%  difference {res['difference'][k]['value']:+.1f} {res['difference'][k]['interval']}")
print("leader statements:", base, f"{point[base]['leader_right']}/{point[base]['leader_said']}", new, f"{point[new]['leader_right']}/{point[new]['leader_said']}", "| races", len(races))
out.write_text(json.dumps(res, indent=1))
