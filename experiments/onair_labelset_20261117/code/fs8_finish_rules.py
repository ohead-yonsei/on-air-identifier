"""POST-HOC / development: finish-stretch statement rules on the DA boxes. usage: fs8_finish_rules.py DIR [DIR ...]
Rules differ in: order (y = lowest box first, x = rightmost box first), front K, votes needed, and whether boxes in the
upper part of the picture (box bottom above YMIN) are ignored."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
CACHE = {}


def labels(O, name, past):
    key = (str(O), name)
    if key not in CACHE:
        rs = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
        out = []
        if rs:
            P = np.array(json.loads((O / "cnn_probs_two" / f"{name}.json").read_text())["probs"])
            Sc = P.copy()
            q = [r["qwen_number"] for r in rs]
            for i, n in enumerate(q):
                if n:
                    Sc[i, n - 1] += 1.0
            ri, cj = linear_sum_assignment(-Sc)
            a = {int(i): int(j) for i, j in zip(ri, cj)}
            out = [(rs[i]["box"], a[i] + 1, bool(q[i] == a[i] + 1 or P[i, a[i]] >= 0.8)) for i in range(len(rs)) if i in a]
        CACHE[key] = out
    return CACHE[key]


def score(dirs, rule, K, m_min, ymin):
    n = st = ok = good = 0
    for d in dirs:
        O = ROOT / d
        man = json.loads((O / "manifest.json").read_text())
        order = json.loads((O / "finish_order.json").read_text())["order"]
        for m in man["frames"]:
            if m["race"] not in order:
                continue
            T = order[m["race"]][:3]
            votes = Counter()
            for name, past in ((m["frame"], False), (f"{m['frame']}_m05", True), (f"{m['frame']}_m10", True)):
                L = [r for r in labels(O, name, past) if r[0][3] >= ymin]
                L = sorted(L, key=(lambda r: -r[0][3]) if rule == "y" else (lambda r: -(r[0][0] + r[0][2])))[:K]
                for num in {num for _, num, s in L if s}:
                    votes[num] += 1
            said = [num for num, v in votes.items() if v >= m_min]
            n += 1
            st += len(said)
            ok += sum(x in T for x in said)
            good += any(x in T for x in said)
    return n, st, ok, good


sets = [[d] for d in sys.argv[1:]] + [sys.argv[1:]]
for rule in ("y", "x"):
    for K in (1, 2, 3):
        for m_min in (1, 2):
            for ymin in (0, 450):
                parts = []
                for ds in sets:
                    n, st, ok, good = score(ds, rule, K, m_min, ymin)
                    parts.append(f"A {ok:2d}/{st:2d}={ok / max(1, st):.2f} B {good:2d}/{n}={good / n:.2f}")
                print(f"order {rule} K{K} votes>={m_min} ymin {ymin:3d} | " + " | ".join(parts))
