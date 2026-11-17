"""Compare running-order rules against the broadcaster's strip. See order/DESIGN_KO.md.

usage: python p1_order_rules.py dev            -> scores every rule on development races and picks one
       python p1_order_rules.py test RULE      -> compares RULE with V on held-out races
Uses stripset/out/D0 (riders, numbers, sure flags) and stripset/det (horse boxes). No model calls.
"""
import json
import statistics
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "stripset"
split = sys.argv[1]
RULES = ["V", "X", "S0.05", "S0.10", "S0.15", "S0.20"]
frames = [m for m in json.loads((S / "manifest.json").read_text())["frames"] if m["split"] == split]


def lean_of(riders, horses):
    """Median of (rider centre x - horse centre x) / horse width over riders matched to a horse."""
    vals = []
    for r in riders:
        b = r["box"]
        area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        best, bi = 0.0, None
        for h in horses:
            inter = max(0, min(b[2], h[2]) - max(b[0], h[0])) * max(0, min(b[3], h[3]) - max(b[1], h[1]))
            if inter / area > best:
                best, bi = inter / area, h
        if bi is not None and best >= 0.15:
            vals.append((((b[0] + b[2]) / 2) - ((bi[0] + bi[2]) / 2)) / max(1.0, bi[2] - bi[0]))
    return statistics.median(vals) if len(vals) >= 2 else 0.0


def key_fn(rule, lean):
    """Returns a function: larger value = further ahead."""
    vertical = lambda r: r["box"][3]
    if rule == "V" or lean == 0:
        return vertical
    sign = 1.0 if lean > 0 else -1.0
    horizontal = lambda r: sign * (r["box"][0] + r["box"][2]) / 2
    if rule == "X":
        return horizontal
    return horizontal if abs(lean) >= float(rule[1:]) else vertical


data = []
for m in frames:
    riders = json.loads((S / "out" / "D0" / f"{m['frame']}.json").read_text())["riders"]
    det = json.loads((S / "det" / f"{m['frame']}.json").read_text())
    data.append((m, riders, lean_of(riders, det["horses"])))


def evaluate(rule):
    k = {"pairs": 0, "pairs_agree": 0, "front3_sure": 0, "front3_sure_top3": 0, "front3_sure_top5": 0, "frames": 0,
         "frames_with_sure_top3_in_front3": 0, "lead_sure": 0, "lead_sure_is_first": 0, "lead_sure_in_top3": 0, "horizontal_frames": 0}
    pair_ok = {}
    for m, riders, lean in data:
        B = m["strip"]
        rank = {n: i for i, n in enumerate(B)}
        f = key_fn(rule, lean)
        k["frames"] += 1
        k["horizontal_frames"] += f(riders[0]) != riders[0]["box"][3] if riders else 0
        sure = [r for r in riders if r["state"] == "sure" and r["no"] in rank]
        for i in range(len(sure)):
            for j in range(i + 1, len(sure)):
                a, b = sure[i], sure[j]
                if a["no"] == b["no"]:
                    continue
                ok = (f(a) > f(b)) == (rank[a["no"]] < rank[b["no"]])
                k["pairs"] += 1
                k["pairs_agree"] += ok
                pair_ok[(m["frame"], min(a["no"], b["no"]), max(a["no"], b["no"]))] = ok
        front = sorted(riders, key=lambda r: -f(r))[:3]
        good = False
        for r in front:
            if r["state"] == "sure":
                k["front3_sure"] += 1
                k["front3_sure_top3"] += r["no"] in B[:3]
                k["front3_sure_top5"] += r["no"] in B
                good = good or r["no"] in B[:3]
        k["frames_with_sure_top3_in_front3"] += good
        if front and front[0]["state"] == "sure":
            k["lead_sure"] += 1
            k["lead_sure_is_first"] += front[0]["no"] == B[0]
            k["lead_sure_in_top3"] += front[0]["no"] in B[:3]
    return k, pair_ok


res = {r: evaluate(r) for r in RULES}
rate = lambda a, b: a / b if b else 0.0
leans = [abs(l) for _, _, l in data]
print(f"split {split}: {len(frames)} frames | |lean| quartiles {[round(x, 3) for x in statistics.quantiles(leans, n=4)]}")
print("rule    pairs-agree        A: front3-sure in top3 (top5)      B: frames w/ sure top3 in front3   lead-sure =#1 / in top3   horizontal frames")
for r in RULES:
    k = res[r][0]
    print(f"{r:6s}  {k['pairs_agree']:3d}/{k['pairs']:3d} ({rate(k['pairs_agree'], k['pairs']):.3f})   "
          f"{k['front3_sure_top3']:3d}/{k['front3_sure']:3d} ({rate(k['front3_sure_top3'], k['front3_sure']):.3f}) ({rate(k['front3_sure_top5'], k['front3_sure']):.3f})        "
          f"{k['frames_with_sure_top3_in_front3']:2d}/{k['frames']} ({rate(k['frames_with_sure_top3_in_front3'], k['frames']):.3f})                 "
          f"{k['lead_sure_is_first']}/{k['lead_sure']} , {k['lead_sure_in_top3']}/{k['lead_sure']}          {k['horizontal_frames']}")
out = {"split": split, "frames": len(frames), "rules": {r: res[r][0] for r in RULES}}
if split == "dev":
    out["chosen"] = max(RULES, key=lambda r: (round(rate(res[r][0]["pairs_agree"], res[r][0]["pairs"]), 6), -RULES.index(r)))
    print("chosen on dev:", out["chosen"])
else:
    rule = sys.argv[2]
    a, b = res["V"][1], res[rule][1]
    keys = set(a) & set(b)
    imp, wor = sum(b[x] and not a[x] for x in keys), sum(a[x] and not b[x] for x in keys)
    n, mm = imp + wor, min(imp, wor)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(mm + 1)) / 2 ** n) if n else 1.0
    out["confirm"] = {"rule": rule, "improved": imp, "worsened": wor, "sign_test_p": p, "better": bool(imp > wor and p < 0.05)}
    print(f"V -> {rule}: improved {imp} / worsened {wor}  p={p:.4f}  better: {out['confirm']['better']}")
(ROOT / "order").mkdir(exist_ok=True)
(ROOT / "order" / f"score_{split}.json").write_text(json.dumps(out, indent=1))
