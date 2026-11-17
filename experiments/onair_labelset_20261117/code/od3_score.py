"""Other-day test, step 3: A and B against the strip for the old identifier and the new one (C2). See otherday/DESIGN_KO.md."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "otherday"
man = json.loads((O / "manifest.json").read_text())
frames = man["frames"]


def load(name, past):
    riders = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())
    P = json.loads((O / "cnn_probs" / f"{name}.json").read_text())["probs"]
    return riders, np.array(P) if P else np.zeros((0, 11))


def label(riders, P, variant):
    riders = riders["riders"]
    if variant == "OLD":
        return [(r["box"][3], r["no"], r["state"] == "sure") for r in riders]
    if not riders:
        return []
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


res = {}
stats = Counter()
for variant in ("OLD", "C2"):
    st = ok = ok5 = 0
    by_race = defaultdict(lambda: Counter())
    good = 0
    for m in frames:
        B = m["strip"]
        votes = Counter()
        for name, past in ((m["frame"], False), (f"{m['frame']}_m05", True), (f"{m['frame']}_m10", True)):
            riders, P = load(name, past)
            lab = label(riders, P, variant)
            if variant == "C2" and not past:
                stats["frames"] += 1
                stats["riders"] += len(lab)
                stats["sure"] += sum(s for _, _, s in lab)
                stats["qwen_failures"] += riders.get("qwen_failures", 0)
            for n in {n for _, n, s in sorted(lab, key=lambda x: -x[0])[:2] if s and n}:
                votes[n] += 1
        said = [n for n, v in votes.items() if v >= 2]
        st += len(said)
        c3 = sum(n in B[:3] for n in said)
        ok += c3
        ok5 += sum(n in B for n in said)
        g = any(n in B[:3] for n in said)
        good += g
        r = by_race[m["race"]]
        r["frames"] += 1
        r["statements"] += len(said)
        r["in_top3"] += c3
        r["good"] += g
    res[variant] = {"statements": st, "in_top3": ok, "A": ok / st if st else 0.0, "A_top5": ok5 / st if st else 0.0, "B": good / len(frames) if frames else 0.0,
                    "by_race": {k: dict(v) for k, v in sorted(by_race.items())}}
new = res["C2"]
verdict = {"A_at_least_85": bool(new["A"] >= 0.85), "B_at_least_60": bool(new["B"] >= 0.60), "goal_A_90": bool(new["A"] >= 0.90), "goal_B_80": bool(new["B"] >= 0.80)}
out = {"day": man["day"], "frames": len(frames), "results": res, "verdict": verdict,
       "riders_per_frame": stats["riders"] / max(1, stats["frames"]), "sure_share": stats["sure"] / max(1, stats["riders"]), "qwen_failures": stats["qwen_failures"],
       "strip_numbers_above_11": sum(any(n > 11 for n in m["strip"]) for m in frames)}
(O / "score.json").write_text(json.dumps(out, indent=1))
print(f"{man['day']}: {len(frames)} frames | riders per frame {out['riders_per_frame']:.1f} | sure share {out['sure_share']:.3f} | frames whose strip has a number above 11: {out['strip_numbers_above_11']}")
for v in ("OLD", "C2"):
    r = res[v]
    print(f"  {v:3s} statements {r['statements']:4d}  A {r['A']:.3f} (top5 {r['A_top5']:.3f})  B {r['B']:.3f}")
print("  by race (C2): " + "  ".join(f"{k}: A {v['in_top3']}/{v['statements']} B {v['good']}/{v['frames']}" for k, v in res["C2"]["by_race"].items()))
print("verdict:", verdict)
