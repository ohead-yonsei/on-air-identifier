"""Finish stretch, development scoring (not a test). usage: fs6_dev_finish.py DIR [DIR ...]   e.g. finish_D3/otherday2 finish/otherday2
Grid: classifier folder, order rule (y = lower box is ahead, x = box further right is ahead), front K boxes, sure rule, votes needed."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
CACHE = {}


def labels(O, folder, name, past):
    key = (str(O), folder, name)
    if key not in CACHE:
        rs = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
        out = []
        if rs:
            P = np.array(json.loads((O / folder / f"{name}.json").read_text())["probs"])
            Sc = P.copy()
            q = [r["qwen_number"] for r in rs]
            for i, n in enumerate(q):
                if n:
                    Sc[i, n - 1] += 1.0
            ri, cj = linear_sum_assignment(-Sc)
            a = {int(i): int(j) for i, j in zip(ri, cj)}
            out = [(rs[i]["box"], a[i] + 1, float(P[i, a[i]]), q[i] == a[i] + 1) for i in range(len(rs)) if i in a]
        CACHE[key] = out
    return CACHE[key]


def score(O, folder, order_rule, K, mode, thr, m_min):
    man = json.loads((O / "manifest.json").read_text())
    order = json.loads((O / "finish_order.json").read_text())["order"]
    frames = [m for m in man["frames"] if m["race"] in order]
    st = ok = good = 0
    for m in frames:
        T = order[m["race"]][:3]
        votes = Counter()
        for name, past in ((m["frame"], False), (f"{m['frame']}_m05", True), (f"{m['frame']}_m10", True)):
            L = labels(O, folder, name, past)
            L = sorted(L, key=(lambda r: -r[0][3]) if order_rule == "y" else (lambda r: -(r[0][0] + r[0][2]) / 2))[:K]
            for n in {n for _, n, p, ag in L if ((ag or p >= thr) if mode == "C2" else p >= thr)}:
                votes[n] += 1
        said = [n for n, v in votes.items() if v >= m_min]
        st += len(said)
        ok += sum(n in T for n in said)
        good += any(n in T for n in said)
    return len(frames), st, ok, good


for d in sys.argv[1:]:
    O = ROOT / d
    man = json.loads((O / "manifest.json").read_text())
    nb = [len(json.loads((O / "out" / f"{m['frame']}.json").read_text())["riders"]) for m in man["frames"]]
    print(f"== {d}: boxes per frame mean {sum(nb) / len(nb):.1f}, frames with none {sum(n == 0 for n in nb)}/{len(nb)}")
    rows = []
    for folder in ("cnn_probs_two", "cnn_probs_weak"):
        for order_rule in ("y", "x"):
            for K in (1, 2, 3):
                for mode, thr in (("C2", 0.8), ("CONF", 0.8), ("CONF", 0.6)):
                    for m_min in (1, 2):
                        n, st, ok, good = score(O, folder, order_rule, K, mode, thr, m_min)
                        rows.append((good / n, ok / st if st else 0, st, folder, order_rule, K, mode, thr, m_min))
    for r in sorted(rows, reverse=True)[:14]:
        print(f"  B {r[0]:.3f} A {r[1]:.3f} n {r[2]:3d} | {r[3]} order {r[4]} K {r[5]} {r[6]} {r[7]} votes>={r[8]}")
    print("  reference (two, y, K2, C2 0.8, 2):", [f"{x:.3f}" if isinstance(x, float) else x for x in next(r for r in rows if r[3:] == ("cnn_probs_two", "y", 2, "C2", 0.8, 2))[:3]])
    print("  x-order (two, x, K2, C2 0.8, 2):", [f"{x:.3f}" if isinstance(x, float) else x for x in next(r for r in rows if r[3:] == ("cnn_probs_two", "x", 2, "C2", 0.8, 2))[:3]])
