"""v12: does the program (identify_l1.py --v12) give what the replay (code/v12_rule.py) gives? usage: v12_check.py DAY RACE IDENTIFY_OUT.json
The program is run with --start (video finish - 3 J) --end (video finish) --step 3, so its time points are the replay's; the replay uses the
models of all development days, as the program does. Pictures can differ by one frame (the program takes t-0.5 from the frame times)."""
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "identifier"))
import fn_common as fc
import v12_rule as R
from identify_l1 import V12_CARRY, V12_K, V12_TAU

day, race, prog = sys.argv[1], sys.argv[2], json.loads(Path(sys.argv[3]).read_text())
dr = json.loads((ROOT / "v12" / "dense" / f"{day}.json").read_text())["races"][race]
pts = R.race_points(dr, fc.official(day)[race], joblib.load(R.V11_MODEL), joblib.load(ROOT / "v12" / "models" / "top5_all.joblib"),
                    joblib.load(ROOT / "v12" / "models" / "gap_all.joblib"))
sim = R.run_v12(pts, V12_TAU, V12_K, carry=V12_CARRY)
v11 = R.run_v11(pts)
by_tf = {round(p["tf"], 1): (s, w) for p, s, w in zip(pts, sim, v11)}
same_set = same_order = same_v11 = n = 0
for p in prog["points"]:
    tf = round(dr["video_finish"] - p["asked_t"], 1)
    if tf not in by_tf:
        continue
    s, w = by_tf[tf]
    a, b = [x["number"] for x in p["group5"]], [x for x, _ in s]
    n += 1
    same_set += set(a) == set(b)
    same_order += a == b
    same_v11 += sorted(p["statements"]) == sorted(w)
    if a != b or sorted(p["statements"]) != sorted(w):
        print(f"tf {tf:5.1f} program {a} replay {b} | v11 program {p['statements']} replay {w}")
print({"points": n, "same named set": same_set, "same order": same_order, "same v11 statements": same_v11})
