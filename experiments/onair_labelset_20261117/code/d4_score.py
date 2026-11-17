"""Fourth-day test (2026-10-24), step 3: score four systems against the strip. See weak/DESIGN_KO.md.

S0 = two-day classifier, sure rule C2 (the system so far)
S1 = two-day classifier, sure rule CONF (classifier probability >= 0.8 only)
S2 = strip-trained classifier, sure rule CONF
S3 = strip-trained classifier, sure rule C2 (reference)
S1b, S2b = as S1, S2 with probability >= 0.9 (reported only)
Statement rule for all: a number that is sure among the two front boxes in at least 2 of the 3 most recent frames."""
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "otherday3"
man = json.loads((O / "manifest.json").read_text())
frames = man["frames"]
SYSTEMS = {"S0": ("cnn_probs_two", "C2", 0.8), "S1": ("cnn_probs_two", "CONF", 0.8), "S2": ("cnn_probs_weak", "CONF", 0.8), "S3": ("cnn_probs_weak", "C2", 0.8),
           "S1b": ("cnn_probs_two", "CONF", 0.9), "S2b": ("cnn_probs_weak", "CONF", 0.9)}   # S1b, S2b: reported only, not part of the verdict


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


res, good = {}, {}
for name_s, (folder, mode, THR) in SYSTEMS.items():
    st = ok = ok5 = 0
    by_race = defaultdict(Counter)
    good[name_s] = {}
    for m in frames:
        B = m["strip"]
        votes = Counter()
        for name, past in ((m["frame"], False), (f"{m['frame']}_m05", True), (f"{m['frame']}_m10", True)):
            L = sorted(labels(folder, name, past), key=lambda x: -x[0])[:2]
            for n in {n for _, n, p, ag in L if ((ag or p >= THR) if mode == "C2" else p >= THR)}:
                votes[n] += 1
        said = [n for n, v in votes.items() if v >= 2]
        c3 = sum(n in B[:3] for n in said)
        st += len(said)
        ok += c3
        ok5 += sum(n in B for n in said)
        g = any(n in B[:3] for n in said)
        good[name_s][m["frame"]] = g
        r = by_race[m["race"]]
        r["frames"] += 1
        r["statements"] += len(said)
        r["in_top3"] += c3
        r["good"] += g
    res[name_s] = {"classifier": folder, "sure_rule": mode, "threshold": THR, "statements": st, "in_top3": ok, "A": ok / st if st else 0.0, "A_top5": ok5 / st if st else 0.0,
                   "B": sum(good[name_s].values()) / len(frames) if frames else 0.0, "by_race": {k: dict(v) for k, v in sorted(by_race.items())}}


def paired(a, b):
    imp = sum(good[b][f] and not good[a][f] for f in good[a])
    wor = sum(good[a][f] and not good[b][f] for f in good[a])
    n, mm = imp + wor, min(imp, wor)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(mm + 1)) / 2 ** n) if n else 1.0
    return imp, wor, p


i21, w21, p21 = paired("S1", "S2")
i10, w10, p10 = paired("S0", "S1")
i20, w20, p20 = paired("S0", "S2")
verdict = {
    "V1_rule_raises_A": bool(res["S1"]["A"] >= res["S0"]["A"] + 0.03 and res["S1"]["A"] >= 0.85),
    "V1_A_S0": res["S0"]["A"], "V1_A_S1": res["S1"]["A"], "V1_B_S0": res["S0"]["B"], "V1_B_S1": res["S1"]["B"],
    "V1_frames_S0_to_S1": {"improved": i10, "worsened": w10, "p": p10},
    "V2_strip_training_better": bool(i21 > w21 and p21 < 0.05 and res["S2"]["A"] >= res["S1"]["A"] - 0.02),
    "V2_frames_S1_to_S2": {"improved": i21, "worsened": w21, "p": p21}, "V2_A_change": res["S2"]["A"] - res["S1"]["A"],
    "frames_S0_to_S2": {"improved": i20, "worsened": w20, "p": p20},
    "S2_A_at_least_85": bool(res["S2"]["A"] >= 0.85), "S2_goal_A_90": bool(res["S2"]["A"] >= 0.90),
    "S2_B_at_least_60": bool(res["S2"]["B"] >= 0.60), "S2_goal_B_80": bool(res["S2"]["B"] >= 0.80)}
out = {"day": man["day"], "frames": len(frames), "results": res, "verdict": verdict,
       "strip_numbers_above_11": sum(any(x > 11 for x in m["strip"]) for m in frames)}
(O / "score.json").write_text(json.dumps(out, indent=1))
print(f"{man['day']}: {len(frames)} frames | frames whose strip has a number above 11: {out['strip_numbers_above_11']}")
for v, r in res.items():
    print(f"  {v:3s} ({r['classifier']}, {r['sure_rule']} {r['threshold']}): statements {r['statements']:4d}  A {r['A']:.3f} (top5 {r['A_top5']:.3f})  B {r['B']:.3f}")
for v in ("S0", "S2"):
    print(f"  by race ({v}): " + "  ".join(f"{k}: A {x['in_top3']}/{x['statements']} B {x['good']}/{x['frames']}" for k, x in res[v]["by_race"].items()))
print("verdict:", json.dumps(verdict))
