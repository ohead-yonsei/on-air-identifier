"""Score adaptive detection (DA) against the usual detection (D0), same classifier and rules (S0). See da/DESIGN_KO.md.
usage: da2_score.py strip|finish D0_DIR DA_DIR [D0_DIR DA_DIR ...]     (pairs are pooled; per-pair lines are printed too)
strip: reference = strip top three (manifest "strip").  finish: reference = first three across the line (finish_order.json)."""
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
kind, dirs = sys.argv[1], sys.argv[2:]
pairs = [(dirs[i], dirs[i + 1]) for i in range(0, len(dirs), 2)]
FOLDER, THR = "cnn_probs_two", 0.8


def labels(O, name, past):
    rs = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
    if not rs:
        return []
    P = np.array(json.loads((O / FOLDER / f"{name}.json").read_text())["probs"])
    Sc = P.copy()
    q = [r["qwen_number"] for r in rs]
    for i, n in enumerate(q):
        if n:
            Sc[i, n - 1] += 1.0
    ri, cj = linear_sum_assignment(-Sc)
    a = {int(i): int(j) for i, j in zip(ri, cj)}
    return [(rs[i]["box"][3], a[i] + 1, bool(q[i] == a[i] + 1 or P[i, a[i]] >= THR)) for i in range(len(rs)) if i in a]


def run(d):
    O = ROOT / d
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
            for n in {n for _, n, s in sorted(labels(O, name, past), key=lambda x: -x[0])[:2] if s}:
                votes[n] += 1
        said = [n for n, v in votes.items() if v >= 2]
        st += len(said)
        ok += sum(n in T for n in said)
        good[m["frame"]] = any(n in T for n in said)
    return st, ok, good


tot = {"D0": [0, 0, 0, 0], "DA": [0, 0, 0, 0]}
imp = wor = 0
per = []
for d0, da in pairs:
    r0, ra = run(d0), run(da)
    for key, r in (("D0", r0), ("DA", ra)):
        tot[key][0] += r[0]
        tot[key][1] += r[1]
        tot[key][2] += sum(r[2].values())
        tot[key][3] += len(r[2])
    i = sum(ra[2][f] and not r0[2][f] for f in r0[2])
    w = sum(r0[2][f] and not ra[2][f] for f in r0[2])
    imp += i
    wor += w
    per.append({"D0_dir": d0, "DA_dir": da, "frames": len(r0[2]), "D0": {"statements": r0[0], "right": r0[1], "good_frames": sum(r0[2].values())},
                "DA": {"statements": ra[0], "right": ra[1], "good_frames": sum(ra[2].values())}, "improved": i, "worsened": w})
    print(f"{da}: frames {len(r0[2])} | D0 A {r0[1]}/{r0[0]} B {sum(r0[2].values())}/{len(r0[2])} | DA A {ra[1]}/{ra[0]} B {sum(ra[2].values())}/{len(ra[2])} | improved {i} worsened {w}")
n, mm = imp + wor, min(imp, wor)
p = min(1.0, 2 * sum(comb(n, k) for k in range(mm + 1)) / 2 ** n) if n else 1.0
A0, AA = (tot[k][1] / tot[k][0] if tot[k][0] else 0.0 for k in ("D0", "DA"))
B0, BA = (tot[k][2] / tot[k][3] if tot[k][3] else 0.0 for k in ("D0", "DA"))
verdict = {"kind": kind, "frames": tot["D0"][3], "A_D0": A0, "A_DA": AA, "B_D0": B0, "B_DA": BA, "improved": imp, "worsened": wor, "sign_test_p": p,
           "DA_better": bool(imp > wor and p < 0.05 and AA >= A0 - 0.02), "DA_A_at_least_80": bool(AA >= 0.80), "DA_B_at_least_60": bool(BA >= 0.60),
           "goal_A_90": bool(AA >= 0.90), "goal_B_80": bool(BA >= 0.80)}
(ROOT / "da" / f"score_{kind}.json").write_text(json.dumps({"pairs": per, "verdict": verdict}, indent=1))
print("pooled:", json.dumps(verdict))
