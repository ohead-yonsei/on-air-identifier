"""Third-day test, step 3: one-day vs two-day cap classifier inside the identifier, against the strip. See otherday2/DESIGN_KO.md."""
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "otherday2"
man = json.loads((O / "manifest.json").read_text())
frames = man["frames"]
FOLDER = {"ONE_DAY": "cnn_probs", "TWO_DAYS": "cnn_probs2"}


def label(name, past, variant):
    riders = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
    if variant == "OLD":
        return [(r["box"][3], r["no"], r["state"] == "sure") for r in riders]
    if not riders:
        return []
    P = np.array(json.loads((O / FOLDER[variant] / f"{name}.json").read_text())["probs"])
    Sc = P.copy()
    q = [r["qwen_number"] for r in riders]
    for i, n in enumerate(q):
        if n:
            Sc[i, n - 1] += 1.0
    ri, cj = linear_sum_assignment(-Sc)
    a = {int(i): int(j) for i, j in zip(ri, cj)}
    out = []
    for i, r in enumerate(riders):
        j = a.get(i)
        out.append((r["box"][3], 0, False) if j is None else (r["box"][3], j + 1, bool(q[i] == j + 1 or P[i, j] >= 0.8)))
    return out


res, good = {}, {}
for variant in ("OLD", "ONE_DAY", "TWO_DAYS"):
    st = ok = ok5 = 0
    by_race = defaultdict(Counter)
    good[variant] = {}
    for m in frames:
        B = m["strip"]
        votes = Counter()
        for name, past in ((m["frame"], False), (f"{m['frame']}_m05", True), (f"{m['frame']}_m10", True)):
            for n in {n for _, n, s in sorted(label(name, past, variant), key=lambda x: -x[0])[:2] if s and n}:
                votes[n] += 1
        said = [n for n, v in votes.items() if v >= 2]
        c3 = sum(n in B[:3] for n in said)
        st += len(said)
        ok += c3
        ok5 += sum(n in B for n in said)
        g = any(n in B[:3] for n in said)
        good[variant][m["frame"]] = g
        r = by_race[m["race"]]
        r["frames"] += 1
        r["statements"] += len(said)
        r["in_top3"] += c3
        r["good"] += g
    res[variant] = {"statements": st, "in_top3": ok, "A": ok / st if st else 0.0, "A_top5": ok5 / st if st else 0.0,
                    "B": sum(good[variant].values()) / len(frames) if frames else 0.0, "by_race": {k: dict(v) for k, v in sorted(by_race.items())}}
imp = sum(good["TWO_DAYS"][f] and not good["ONE_DAY"][f] for f in good["ONE_DAY"])
wor = sum(good["ONE_DAY"][f] and not good["TWO_DAYS"][f] for f in good["ONE_DAY"])
n, mm = imp + wor, min(imp, wor)
p = min(1.0, 2 * sum(comb(n, i) for i in range(mm + 1)) / 2 ** n) if n else 1.0
two, one = res["TWO_DAYS"], res["ONE_DAY"]
verdict = {"two_days_better": bool(imp > wor and p < 0.05 and two["A"] >= one["A"] - 0.02), "improved_frames": imp, "worsened_frames": wor, "sign_test_p": p,
           "A_change": two["A"] - one["A"], "A_at_least_85": bool(two["A"] >= 0.85), "goal_A_90": bool(two["A"] >= 0.90),
           "B_at_least_60": bool(two["B"] >= 0.60), "goal_B_80": bool(two["B"] >= 0.80)}
out = {"day": man["day"], "frames": len(frames), "results": res, "verdict": verdict,
       "strip_numbers_above_11": sum(any(x > 11 for x in m["strip"]) for m in frames)}
(O / "score.json").write_text(json.dumps(out, indent=1))
print(f"{man['day']}: {len(frames)} frames | frames whose strip has a number above 11: {out['strip_numbers_above_11']}")
for v in ("OLD", "ONE_DAY", "TWO_DAYS"):
    r = res[v]
    print(f"  {v:8s} statements {r['statements']:4d}  A {r['A']:.3f} (top5 {r['A_top5']:.3f})  B {r['B']:.3f}")
print("  by race (TWO_DAYS): " + "  ".join(f"{k}: A {v['in_top3']}/{v['statements']} B {v['good']}/{v['frames']}" for k, v in two["by_race"].items()))
print("  by race (ONE_DAY) : " + "  ".join(f"{k}: A {v['in_top3']}/{v['statements']} B {v['good']}/{v['frames']}" for k, v in one["by_race"].items()))
print("verdict:", verdict)
