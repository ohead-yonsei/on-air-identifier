"""Compare two outputs of identifier/identify_l1.py time point by time point (statements, leader, order, fallback numbers).
usage: id_same.py NEW.json OLD.json      only the time points present in NEW are compared (NEW may cover fewer races)"""
import json
import sys
from pathlib import Path

new, old = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
by = {p["frame"]: p for p in old["points"]}


def key(p):
    fb = sorted(d["number"] for d in p.get("detail", []) if d.get("source") == "fallback")
    return sorted(p["statements"]), p.get("leader"), sorted((o["ahead"], o["behind"]) for o in p.get("order", [])), fb


n = same = missing = 0
for p in new["points"]:
    q = by.get(p["frame"])
    if q is None:
        missing += 1
        continue
    n += 1
    if key(p) == key(q):
        same += 1
    else:
        print("differs", p["frame"], key(p), key(q))
print(f"configuration new {new.get('configuration')} (version {new.get('version')}) old {old.get('configuration')}; "
      f"time points compared {n}; identical {same}; not in old {missing}")
