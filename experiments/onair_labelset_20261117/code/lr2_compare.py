"""Compare two cap-classifier folders on the same boxes, same rules (S0). Development use unless a design says otherwise.
usage: lr2_compare.py strip|finish FOLDER_A FOLDER_B DIR [DIR ...]"""
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
kind, fa, fb = sys.argv[1:4]


def labels(O, folder, name, past):
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
    return [(rs[i]["box"][3], a[i] + 1, bool(q[i] == a[i] + 1 or P[i, a[i]] >= 0.8)) for i in range(len(rs)) if i in a]


def run(O, folder):
    man = json.loads((O / "manifest.json").read_text())
    if kind == "finish":
        order = json.loads((O / "finish_order.json").read_text())["order"]
        frames = [(m, order[m["race"]][:3]) for m in man["frames"] if m["race"] in order]
    else:
        frames = [(m, m["strip"][:3]) for m in man["frames"]]
    st = ok = 0
    good = {}
    for m, T in frames:
        votes = Counter()
        for name, past in ((m["frame"], False), (f"{m['frame']}_m05", True), (f"{m['frame']}_m10", True)):
            for n in {n for _, n, s in sorted(labels(O, folder, name, past), key=lambda x: -x[0])[:2] if s}:
                votes[n] += 1
        said = [n for n, v in votes.items() if v >= 2]
        st += len(said)
        ok += sum(n in T for n in said)
        good[m["frame"]] = any(n in T for n in said)
    return st, ok, good


T = {fa: [0, 0, 0, 0], fb: [0, 0, 0, 0]}
imp = wor = 0
for d in sys.argv[4:]:
    a, b = run(ROOT / d, fa), run(ROOT / d, fb)
    for k, r in ((fa, a), (fb, b)):
        T[k][0] += r[0]; T[k][1] += r[1]; T[k][2] += sum(r[2].values()); T[k][3] += len(r[2])
    i = sum(b[2][f] and not a[2][f] for f in a[2]); w = sum(a[2][f] and not b[2][f] for f in a[2])
    imp += i; wor += w
    print(f"{d}: {fa} A {a[1]}/{a[0]} B {sum(a[2].values())}/{len(a[2])} | {fb} A {b[1]}/{b[0]} B {sum(b[2].values())}/{len(b[2])} | improved {i} worsened {w}")
n, mm = imp + wor, min(imp, wor)
p = min(1.0, 2 * sum(comb(n, k) for k in range(mm + 1)) / 2 ** n) if n else 1.0
out = {"kind": kind, "frames": T[fa][3], "A_a": T[fa][1] / max(1, T[fa][0]), "A_b": T[fb][1] / max(1, T[fb][0]), "B_a": T[fa][2] / max(1, T[fa][3]), "B_b": T[fb][2] / max(1, T[fb][3]),
       "improved": imp, "worsened": wor, "sign_test_p": p}
print("pooled:", json.dumps(out))
