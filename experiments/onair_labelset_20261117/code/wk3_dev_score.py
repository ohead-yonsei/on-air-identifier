"""Development scoring of cap classifiers on a day already seen. usage: wk3_dev_score.py DAY_DIR FOLDER_A FOLDER_B
Prints A and B for a few sure rules, and frames improved / worsened (B vs A) per rule. Not a test."""
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
day, fa, fb = sys.argv[1:4]
O = ROOT / day
frames = json.loads((O / "manifest.json").read_text())["frames"]


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


def score(folder, mode, thr):
    st = ok = 0
    good = {}
    for m in frames:
        B = m["strip"]
        votes = Counter()
        for name, past in ((m["frame"], False), (f"{m['frame']}_m05", True), (f"{m['frame']}_m10", True)):
            L = sorted(labels(folder, name, past), key=lambda x: -x[0])[:2]
            for n in {n for _, n, p, ag in L if ((ag or p >= thr) if mode == "C2" else p >= thr)}:
                votes[n] += 1
        said = [n for n, v in votes.items() if v >= 2]
        st += len(said)
        ok += sum(n in B[:3] for n in said)
        good[m["frame"]] = any(n in B[:3] for n in said)
    return st, ok / st if st else 0.0, sum(good.values()) / len(frames), good


for mode, thr in (("C2", 0.8), ("CONF", 0.7), ("CONF", 0.8), ("CONF", 0.9)):
    a, b = score(fa, mode, thr), score(fb, mode, thr)
    imp = sum(b[3][f] and not a[3][f] for f in a[3])
    wor = sum(a[3][f] and not b[3][f] for f in a[3])
    n, mm = imp + wor, min(imp, wor)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(mm + 1)) / 2 ** n) if n else 1.0
    print(f"{day} {mode} {thr}: {fa}: n {a[0]} A {a[1]:.3f} B {a[2]:.3f} | {fb}: n {b[0]} A {b[1]:.3f} B {b[2]:.3f} | improved {imp} worsened {wor} p {p:.3f}")
