"""Confirmation day, finish stretch: usual detection + vertical order (BASE) against adaptive detection + horizontal order (FIN_X).
usage: cf_score_finish.py DAY_DIR      reads finish/DAY_DIR (D0) and da/finish_DAY_DIR (DA). See confirm/DESIGN_KO.md."""
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
day = sys.argv[1]
SYSTEMS = {"BASE": (ROOT / "finish" / day, "y"), "FIN_Y": (ROOT / "da" / f"finish_{day}", "y"), "FIN_X": (ROOT / "da" / f"finish_{day}", "x")}


def labels(O, name, past):
    rs = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
    if not rs:
        return []
    P = np.array(json.loads((O / "cnn_probs_two" / f"{name}.json").read_text())["probs"])
    Sc = P.copy()
    q = [r["qwen_number"] for r in rs]
    for i, n in enumerate(q):
        if n:
            Sc[i, n - 1] += 1.0
    ri, cj = linear_sum_assignment(-Sc)
    a = {int(i): int(j) for i, j in zip(ri, cj)}
    return [(rs[i]["box"], a[i] + 1, bool(q[i] == a[i] + 1 or P[i, a[i]] >= 0.8)) for i in range(len(rs)) if i in a]


res, good = {}, {}
for s, (O, rule) in SYSTEMS.items():
    man = json.loads((O / "manifest.json").read_text())
    order = json.loads((O / "finish_order.json").read_text())["order"]
    st = ok = 0
    good[s] = {}
    by = {}
    for m in man["frames"]:
        if m["race"] not in order:
            continue
        T = order[m["race"]][:3]
        votes = Counter()
        for name, past in ((m["frame"], False), (f"{m['frame']}_m05", True), (f"{m['frame']}_m10", True)):
            L = sorted(labels(O, name, past), key=(lambda r: -r[0][3]) if rule == "y" else (lambda r: -(r[0][0] + r[0][2])))[:2]
            for n in {n for _, n, sure in L if sure}:
                votes[n] += 1
        said = [n for n, v in votes.items() if v >= 2]
        st += len(said)
        ok += sum(n in T for n in said)
        good[s][m["frame"]] = any(n in T for n in said)
        b = by.setdefault(str(m["seconds_before_finish"]), [0, 0])
        b[0] += good[s][m["frame"]]
        b[1] += 1
    res[s] = {"frames": len(good[s]), "statements": st, "right": ok, "A": ok / st if st else 0.0, "B": sum(good[s].values()) / max(1, len(good[s])), "by_seconds_before_finish": by}
imp = sum(good["FIN_X"][f] and not good["BASE"][f] for f in good["BASE"])
wor = sum(good["BASE"][f] and not good["FIN_X"][f] for f in good["BASE"])
n, mm = imp + wor, min(imp, wor)
p = min(1.0, 2 * sum(comb(n, k) for k in range(mm + 1)) / 2 ** n) if n else 1.0
x = res["FIN_X"]
verdict = {"improved": imp, "worsened": wor, "sign_test_p": p, "finish_rule_better": bool(imp > wor and p < 0.05 and x["A"] >= 0.80),
           "FIN_X_A_at_least_80": bool(x["A"] >= 0.80), "FIN_X_B_at_least_60": bool(x["B"] >= 0.60), "goal_A_90": bool(x["A"] >= 0.90), "goal_B_80": bool(x["B"] >= 0.80)}
(ROOT / "confirm" / "score_finish.json").write_text(json.dumps({"day": day, "results": res, "verdict": verdict}, indent=1))
for s, r in res.items():
    print(f"{s}: frames {r['frames']} statements {r['statements']} right {r['right']} A {r['A']:.3f} B {r['B']:.3f} | by seconds before finish (good/frames): {r['by_seconds_before_finish']}")
print("verdict:", json.dumps(verdict))
