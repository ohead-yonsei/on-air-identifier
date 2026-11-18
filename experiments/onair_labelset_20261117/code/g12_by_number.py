"""finish3, after the confirmation (results seen; exploration): the leader statement by the number of the TRUE leader.
usage: g12_by_number.py OUT.json            (CPU; v10 as in g11_score2.py, bar 0.7; the fourteen days with whole-race pictures; clock-agrees races)
For every time point: the true leader (official times), whether its number was readable (a sure rider in 2 of the 3 pictures), and what was said.
Also the winner calls (v10's rule) of every day with last-37-s pictures, by the number of the official winner (all races run)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import g5_explore as g5
import g7_winner_guard as g7
import g8_explore2 as g8

DAYS = g5.DEV + g5.HELD + ["otherday14", "otherday15"]
strip = joblib.load(ROOT / "night" / "leader_fast.joblib")
tab = defaultdict(lambda: defaultdict(int))
for d in DAYS:
    fc.FOLDER.setdefault(d, d)
    data = g5.load(d)
    fs = joblib.load(ROOT / "finish2" / "models" / (f"fs_leader_without_{d}.joblib" if d in g5.DEV else "fs_leader_all.joblib"))
    g5.score_day(data, {"strip": (strip, 19), "fs": (fs, 19), "W19": (strip, 19)})
    for race, rr in data.items():
        if not rr["agree"]:
            continue
        for r in g8.race_rows(rr, ("thr", 0.5)):
            ref = rr["pics"][float(r["tau"])]["ref"]["leader"]
            k = str(ref) if ref <= 9 else ("10" if ref == 10 else "11+")
            for kk in (k, "all"):
                t = tab[kk]
                t["points"] += 1
                t["readable"] += int(r["readable"])
                t["right"] += int(r["right"])
                t["wrong"] += int(r["said"] and not r["right"])
res = {"leader_by_true_leader": {k: dict(v) for k, v in tab.items()}}
for k in [str(i) for i in range(1, 10)] + ["10", "11+", "all"]:
    v = tab[k]
    if v["points"]:
        n = v["points"]
        print(f"true leader {k:>3s}: time points {n:5d}  readable {100 * v['readable'] / n:5.1f}%  said right {100 * v['right'] / n:5.1f}%  said wrong {100 * v['wrong'] / n:5.1f}%")
win = defaultdict(lambda: defaultdict(int))
for d in g5.DEV + ["otherday9", "otherday10", "otherday11", "otherday14", "otherday15"]:
    for r in g7.races(d):
        w = next((n for n, x in r["rank"].items() if x == 1), None)
        c = g7.call(r)
        k = "1-9" if w is not None and w <= 9 else "10 or more"
        win[k]["races"] += 1
        win[k]["called"] += int(c is not None)
        win[k]["right"] += int(c is not None and c == w)
res["winner_by_true_winner"] = {k: dict(v) for k, v in win.items()}
for k, v in win.items():
    print(f"official winner {k}: races {v['races']}, called {v['called']}, right {v['right']}")
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
