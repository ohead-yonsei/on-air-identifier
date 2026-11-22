"""live2, post-hoc exploration (not part of the locked verdicts): where in the race the leader lines go wrong.
usage: lv3_explore.py LIVE2_DAY_DIR      standard library only. Bins every point line by seconds before the clock stop."""
import json
import sys
from collections import defaultdict
from pathlib import Path

D = Path(sys.argv[1])
BINS = [(-5, 12, "last 12 s"), (12, 36, "36-12 s before"), (36, 999, "earlier")]
pool = json.loads((D / "pooled.json").read_text())
ok = {r["race"] for r in pool["races"] if r["status"] == "clock agrees"}
tab = defaultdict(lambda: defaultdict(int))
wrong_rows = []
for r in sorted(ok):
    live, sc = json.loads((D / r / "live.json").read_text()), json.loads((D / r / "score.json").read_text())
    fin = live["clock"]["finish"]
    for row in sc["rows"]:
        if "verdict" in row:
            continue
        left = fin - row["video_time"]
        b = next(n for lo, hi, n in BINS if lo <= left < hi)
        t = tab[b]
        t["lines"] += 1
        if row["leader"]:
            t["leader lines"] += 1
            t["leader " + row["leader_verdict"]] += 1
            if row["leader_verdict"] == "wrong":
                wrong_rows.append((r, round(left, 1), row["leader"], row["official_order"][:3]))
        for v in row["numbers"].values():
            t["numbers"] += 1
            t["numbers " + v] += 1
out = {b: dict(tab[b]) for _, _, b in BINS}
for b, t in out.items():
    n = t.get("leader lines", 0)
    print(f"{b:16s} lines {t.get('lines',0):3d}  leader {t.get('leader right',0)}/{n} right, {t.get('leader close',0)} close, {t.get('leader wrong',0)} wrong"
          f"  | numbers {t.get('numbers right',0)}/{t.get('numbers',0)} right")
print("wrong leader lines (race, s before clock stop, said, official first three):")
for w in wrong_rows:
    print(" ", w)
(D.parent / "explore.json").write_text(json.dumps({"bins": out, "wrong_leader_lines": wrong_rows}, ensure_ascii=False, indent=1))
