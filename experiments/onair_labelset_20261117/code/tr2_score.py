"""Sixth-day test: the identifier as it stands (S0) against the time-carrying rule (T1). See temporal/DESIGN_KO.md.
usage: tr2_score.py DAY_DIR        reads DAY_DIR (detection D0) and da/DAY_DIR (adaptive detection DA), classifier folder cnn_probs_two.

S0 : D0, sure = Qwen number kept or classifier probability >= 0.8, statement = sure among the two front boxes in 2 of the 3 most recent pictures.
T1 : DA, sure = classifier probability >= 0.7, same base statement, then: numbers stated now, plus numbers stated at both t-3 s and t-6 s.
T1_D0 : as T1 on D0 boxes (reported only)."""
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
day = sys.argv[1]
SYSTEMS = {"S0": (ROOT / day, "C2", 0.8, False), "T1": (ROOT / "da" / day, "CONF", 0.7, True), "T1_D0": (ROOT / day, "CONF", 0.7, True)}
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
            out = [(rs[i]["box"][3], a[i] + 1, float(P[i, a[i]]), q[i] == a[i] + 1) for i in range(len(rs)) if i in a]
        CACHE[key] = out
    return CACHE[key]


def base(O, frame, mode, thr):
    votes = Counter()
    for name, past in ((frame, False), (f"{frame}_m05", True), (f"{frame}_m10", True)):
        for n in {n for _, n, p, ag in sorted(labels(O, name, past), key=lambda x: -x[0])[:2] if ((ag or p >= thr) if mode == "C2" else p >= thr)}:
            votes[n] += 1
    return {n for n, v in votes.items() if v >= 2}


res, good = {}, {}
for s, (O, mode, thr, carry) in SYSTEMS.items():
    frames = json.loads((O / "manifest.json").read_text())["frames"]
    st = ok = ok5 = 0
    good[s] = {}
    by_race = defaultdict(Counter)
    for m in frames:
        said = base(O, m["frame"], mode, thr)
        if carry:
            race, t = m["frame"].split("_t")
            prev = [base(O, f"{race}_t{int(t) - b:03d}", mode, thr) if (O / "out" / f"{race}_t{int(t) - b:03d}.json").exists() else set() for b in (3, 6)]
            said = said | (prev[0] & prev[1])
        B = m["strip"]
        c3 = sum(n in B[:3] for n in said)
        st += len(said)
        ok += c3
        ok5 += sum(n in B for n in said)
        good[s][m["frame"]] = any(n in B[:3] for n in said)
        r = by_race[m["race"]]
        r["frames"] += 1
        r["statements"] += len(said)
        r["in_top3"] += c3
        r["good"] += good[s][m["frame"]]
    res[s] = {"frames": len(frames), "statements": st, "in_top3": ok, "A": ok / st if st else 0.0, "A_top5": ok5 / st if st else 0.0,
              "B": sum(good[s].values()) / max(1, len(frames)), "by_race": {k: dict(v) for k, v in sorted(by_race.items())}}
imp = sum(good["T1"][f] and not good["S0"][f] for f in good["S0"])
wor = sum(good["S0"][f] and not good["T1"][f] for f in good["S0"])
n, mm = imp + wor, min(imp, wor)
p = min(1.0, 2 * sum(comb(n, k) for k in range(mm + 1)) / 2 ** n) if n else 1.0
s0, t1 = res["S0"], res["T1"]
verdict = {"T1_better": bool(t1["A"] > s0["A"] and t1["B"] > s0["B"] and imp >= wor), "A_change": t1["A"] - s0["A"], "B_change": t1["B"] - s0["B"],
           "improved": imp, "worsened": wor, "sign_test_p": p, "T1_goal_A_90": bool(t1["A"] >= 0.90), "T1_goal_B_80": bool(t1["B"] >= 0.80),
           "S0_goal_A_90": bool(s0["A"] >= 0.90), "S0_goal_B_80": bool(s0["B"] >= 0.80)}
(ROOT / "temporal" / "score.json").write_text(json.dumps({"day": day, "results": res, "verdict": verdict}, indent=1))
for s, r in res.items():
    print(f"{s:6s}: frames {r['frames']} statements {r['statements']} A {r['A']:.3f} (top5 {r['A_top5']:.3f}) B {r['B']:.3f}")
for s in ("S0", "T1"):
    print(f"  by race ({s}): " + "  ".join(f"{k}: A {v['in_top3']}/{v['statements']} B {v['good']}/{v['frames']}" for k, v in res[s]["by_race"].items()))
print("verdict:", json.dumps(verdict))
