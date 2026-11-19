"""bench2: are two whole-race outputs the same, time point by time point? (statements, leader, order)
usage: bw2_same.py NEW.json OLD.json        (standard library only; races missing in NEW are skipped, e.g. races whose clock does not agree)"""
import json
import sys

a, b = (json.load(open(f))["races"] for f in sys.argv[1:3])
n = diff = 0
for race in sorted(a):
    pa, pb = ({round(p["t"], 3): p for p in x[race]["points"]} for x in (a, b))
    if set(pa) != set(pb):
        print(race, "time points differ:", len(pa), len(pb))
        diff += len(set(pa) ^ set(pb))
    for t in sorted(set(pa) & set(pb)):
        n += 1
        if any(pa[t].get(k) != pb[t].get(k) for k in ("statements", "leader", "order")):
            diff += 1
            print(race, t, pa[t]["statements"], pa[t].get("leader"), "|", pb[t]["statements"], pb[t].get("leader"))
print(f"{len(a)} races, {n} time points compared, {diff} different")
