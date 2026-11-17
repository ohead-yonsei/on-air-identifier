"""Seventh-day test: hand rule (T1) against the learned front-of-field rule (L1). See order2/DESIGN_KO.md.
usage: or3_score.py DAY_DIR      reads da/DAY_DIR (adaptive detection, classifier folder cnn_probs_two) and order2/l1_model.joblib."""
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from or_common import ROOT, load_day, statements  # noqa: E402

day = sys.argv[1]
data = load_day(f"da/{day}")
gb = joblib.load(ROOT / "order2" / "l1_model.joblib")
said = {"T1": statements(data, None), "L1": statements(data, lambda F: gb.predict_proba(F)[:, 1])}
res, good = {}, {}
for s in said:
    st = ok = ok5 = 0
    good[s] = {}
    by_race = defaultdict(Counter)
    for m, _ in data:
        nums, B = said[s][m["frame"]], m["strip"]
        c3 = sum(n in B[:3] for n in nums)
        st += len(nums)
        ok += c3
        ok5 += sum(n in B for n in nums)
        good[s][m["frame"]] = any(n in B[:3] for n in nums)
        r = by_race[m["race"]]
        r["frames"] += 1
        r["statements"] += len(nums)
        r["in_top3"] += c3
        r["good"] += good[s][m["frame"]]
    res[s] = {"frames": len(data), "statements": st, "in_top3": ok, "A": ok / st if st else 0.0, "A_top5": ok5 / st if st else 0.0,
              "B": sum(good[s].values()) / max(1, len(data)), "by_race": {k: dict(v) for k, v in sorted(by_race.items())}}
imp = sum(good["L1"][f] and not good["T1"][f] for f in good["T1"])
wor = sum(good["T1"][f] and not good["L1"][f] for f in good["T1"])
n, mm = imp + wor, min(imp, wor)
p = min(1.0, 2 * sum(comb(n, k) for k in range(mm + 1)) / 2 ** n) if n else 1.0
t1, l1 = res["T1"], res["L1"]
verdict = {"L1_meets_both_goals": bool(l1["A"] >= 0.90 and l1["B"] >= 0.80), "L1_goal_A_90": bool(l1["A"] >= 0.90), "L1_goal_B_80": bool(l1["B"] >= 0.80),
           "L1_better_than_T1": bool(l1["B"] > t1["B"] and l1["A"] >= t1["A"] - 0.02 and imp > wor and p < 0.05),
           "A_change": l1["A"] - t1["A"], "B_change": l1["B"] - t1["B"], "improved": imp, "worsened": wor, "sign_test_p": p,
           "T1_goal_A_90": bool(t1["A"] >= 0.90), "T1_goal_B_80": bool(t1["B"] >= 0.80)}
(ROOT / "order2" / "score.json").write_text(json.dumps({"day": day, "results": res, "verdict": verdict}, indent=1))
for s, r in res.items():
    print(f"{s}: frames {r['frames']} statements {r['statements']} A {r['A']:.3f} (top5 {r['A_top5']:.3f}) B {r['B']:.3f}")
    print(f"  by race: " + "  ".join(f"{k}: A {v['in_top3']}/{v['statements']} B {v['good']}/{v['frames']}" for k, v in r["by_race"].items()))
print("verdict:", json.dumps(verdict))
