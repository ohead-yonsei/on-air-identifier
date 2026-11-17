"""Finish stretch, step 4: score the identifier's statements in the last 6 s against the order across the line.
usage: fs4_score.py DAY_DIR      See finish/DESIGN_KO.md.

Reference: finish/DAY_DIR/finish_order.json = {race: [first, second, third, ...]} read from the slow-motion replay.
A = statements that name one of the first three across the line / statements. B = frames with at least one such statement.
Systems as in d4_score.py: S0 (hand-label classifier, C2), S1 (hand-label classifier, CONF 0.8), S2 (strip-trained classifier, CONF 0.8)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
day = sys.argv[1]
O = ROOT / "finish" / day
man = json.loads((O / "manifest.json").read_text())
order = json.loads((O / "finish_order.json").read_text())["order"]
frames = [m for m in man["frames"] if m["race"] in order and len(order[m["race"]]) >= 3]
SYSTEMS = {"S0": ("cnn_probs_two", "C2", 0.8), "S1": ("cnn_probs_two", "CONF", 0.8), "S2": ("cnn_probs_weak", "CONF", 0.8)}


def labels(folder, name, past):
    rs = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
    if not rs:
        return []
    P = np.array(json.loads((O / folder / f"{name}.json").read_text())["probs"])
    Sc = P.copy()
    q = [r["qwen_number"] for r in rs]
    for i, n in enumerate(q):
        if n:
            Sc[i, n - 1] += 1.0
    ri, cj = linear_sum_assignment(-Sc)
    a = {int(i): int(j) for i, j in zip(ri, cj)}
    return [(rs[i]["box"][3], a[i] + 1, float(P[i, a[i]]), q[i] == a[i] + 1) for i in range(len(rs)) if i in a]


res = {}
for name_s, (folder, mode, thr) in SYSTEMS.items():
    st = ok = good = 0
    by_off, by_race = defaultdict(Counter), defaultdict(Counter)
    boxes = 0
    for m in frames:
        T = order[m["race"]][:3]
        votes = Counter()
        for k, (name, past) in enumerate(((m["frame"], False), (f"{m['frame']}_m05", True), (f"{m['frame']}_m10", True))):
            L = labels(folder, name, past)
            if k == 0:
                boxes += len(L)
            for n in {n for _, n, p, ag in sorted(L, key=lambda x: -x[0])[:2] if ((ag or p >= thr) if mode == "C2" else p >= thr)}:
                votes[n] += 1
        said = [n for n, v in votes.items() if v >= 2]
        c = sum(n in T for n in said)
        g = any(n in T for n in said)
        st += len(said)
        ok += c
        good += g
        for key, table in ((str(m["seconds_before_finish"]), by_off), (m["race"], by_race)):
            table[key]["frames"] += 1
            table[key]["statements"] += len(said)
            table[key]["right"] += c
            table[key]["good"] += g
    res[name_s] = {"classifier": folder, "sure_rule": mode, "threshold": thr, "frames": len(frames), "statements": st, "right": ok,
                   "A": ok / st if st else 0.0, "B": good / len(frames) if frames else 0.0, "boxes_per_frame": boxes / len(frames) if frames else 0.0,
                   "by_seconds_before_finish": {k: dict(v) for k, v in sorted(by_off.items())}, "by_race": {k: dict(v) for k, v in sorted(by_race.items())}}
flags = {s: {"A_at_least_90": bool(r["A"] >= 0.90), "B_at_least_80": bool(r["B"] >= 0.80)} for s, r in res.items()}
(O / "score.json").write_text(json.dumps({"day": day, "races": sorted(order), "results": res, "flags": flags}, indent=1))
print(f"finish stretch {day}: {len(frames)} frames from {len({m['race'] for m in frames})} races")
for s, r in res.items():
    print(f"  {s} ({r['classifier']}, {r['sure_rule']} {r['threshold']}): statements {r['statements']:3d} right {r['right']:3d}  A {r['A']:.3f}  B {r['B']:.3f}  boxes/frame {r['boxes_per_frame']:.1f}")
    print("     by seconds before finish: " + "  ".join(f"{k}s: A {v['right']}/{v['statements']} B {v['good']}/{v['frames']}" for k, v in r["by_seconds_before_finish"].items()))
print("flags:", json.dumps(flags))
