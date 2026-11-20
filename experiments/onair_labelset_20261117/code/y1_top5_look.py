"""top5 (exploration, 2026-11-20): what happens when the statement is widened from "in the first three" to "in the first five".
usage: y1_top5_look.py OUT.json DAY [DAY ...]          standard library only, no GPU

Reads what is already stored: bench2/whole/v11/DAY.json (the v11 identifier on the whole race, every 3 s; the riders of the current
picture with their cap number, cap probability and "in the first three" score) and the order from the official passing times
(fn_common.reference_full). Nothing is run again and no new day is used. One picture per time point (the program itself votes over
three pictures and carries numbers; that part is not redone here), so the rows below are a look, not a test.

Parts of the race: start = race time under 18 s, finish = the last 36 s, middle = the rest (roughly where the strip is on screen).
A number 11 from the classifier stands for every cap 11 and above (they look the same); it is given the best-placed such horse."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc

out_path, days = Path(sys.argv[1]), sys.argv[2:]
PARTS = ("all", "start", "middle", "finish")
RULES = [(m, thr) for m in (2, 3, 4, 5) for thr in (0.6, 0.3, 0.1, 0.0)]
pc = lambda a, b: None if not b else round(100 * a / b, 1)

points = []          # one entry per time point
for day in days:
    off = fc.official(day)
    whole = json.loads((ROOT / "bench2" / "whole" / "v11" / f"{day}.json").read_text())["races"]
    for race, rr in whole.items():
        for p in rr["points"]:
            ref = fc.reference_full(off[race], p["race_time"])
            if ref is None or len(ref["order"]) < 6:
                continue
            order, d = ref["order"], ref["to_go"]
            rank = {n: i + 1 for i, n in enumerate(order)}
            high = [n for n in order if n >= 11]
            pos = lambda n, rank=rank, high=high: rank.get(high[0]) if n >= 11 and high else rank.get(n)          # None = no such horse in the race
            gap = lambda n, pos=pos, d=d, order=order: None if pos(n) is None else d[order[pos(n) - 1]]
            best = {}
            for r in p.get("riders_now", []):
                if r.get("sure") and r["number"] and (r["number"] not in best or r["p_lead"] > best[r["number"]]["p_lead"]):
                    best[r["number"]] = r
            sure = sorted(best.values(), key=lambda r: -r["p_lead"])
            part = "start" if p["race_time"] < 18 else ("finish" if p["to_finish"] <= 36 else "middle")
            points.append({"day": day, "race": race, "part": part, "field": len(order), "order": order, "d": d, "pos": pos, "gap": gap,
                           "sure": sure, "said": p.get("statements", []), "riders": len(p.get("riders_now", []))})


def within(pt, n, k, lenient):
    """Is number n in the first k (lenient: or within 3 m of the k-th horse)?"""
    q = pt["pos"](n)
    if q is None:
        return False
    if q <= k:
        return True
    return lenient and pt["gap"](n) <= pt["d"][pt["order"][k - 1]] + fc.MARGIN_M


def rule_row(pts, pick):
    """pick(point) -> numbers said. Counts against the first three and the first five."""
    n_pts = len(pts)
    said = [pick(pt) for pt in pts]
    tot = sum(len(s) for s in said)
    row = {"points": n_pts, "numbers per point": round(tot / n_pts, 2) if n_pts else None, "points with a number": pc(sum(1 for s in said if s), n_pts)}
    for k in (3, 5):
        for len_, tag in ((False, "strict"), (True, "within 3 m")):
            ok = [[within(pt, n, k, len_) for n in s] for pt, s in zip(pts, said)]
            row[f"first {k}, {tag}"] = {
                "numbers right (A)": pc(sum(map(sum, ok)), tot),
                "points with a right number (B)": pc(sum(1 for o in ok if any(o)), n_pts),
                "points with a wrong number": pc(sum(1 for o in ok if not all(o)), n_pts),
                "of the first k, named per point": round(sum(len({pt["pos"](n) for n, o in zip(s, oo) if o and pt["pos"](n) <= k}) for pt, s, oo in zip(pts, said, ok)) / n_pts, 2) if n_pts else None}
    return row


res = {"days": days, "note": __doc__.split("\n\n")[1].replace("\n", " ")}
for part in PARTS:
    pts = [pt for pt in points if part == "all" or pt["part"] == part]
    if not pts:
        continue
    R = {"points": len(pts), "mean field": round(sum(pt["field"] for pt in pts) / len(pts), 1),
         "riders found per picture": round(sum(pt["riders"] for pt in pts) / len(pts), 2),
         "riders with a sure number per picture": round(sum(len(pt["sure"]) for pt in pts) / len(pts), 2)}
    # 1. how often the horse placed k-th has its number read (sure) in the picture
    R["placed k-th: number read"] = {k: pc(sum(1 for pt in pts if any(pt["pos"](r["number"]) == k for r in pt["sure"])), len(pts)) for k in range(1, 9)}
    # 2. where the sure-read riders are placed
    hist = defaultdict(int)
    for pt in pts:
        for r in pt["sure"]:
            q = pt["pos"](r["number"])
            hist["no such horse" if q is None else ("1-3" if q <= 3 else "4-5" if q <= 5 else "6+")] += 1
    tot = sum(hist.values())
    R["sure-read riders by place"] = {k: pc(hist[k], tot) for k in ("1-3", "4-5", "6+", "no such horse")}
    R["chance, a random runner"] = {"first 3": pc(sum(3 / pt["field"] for pt in pts), len(pts)), "first 5": pc(sum(5 / pt["field"] for pt in pts), len(pts))}
    # 3. rules
    R["rules"] = {"v11 as stored (up to 2 numbers, three pictures, carried)": rule_row(pts, lambda pt: pt["said"])}
    for m, thr in RULES:
        R["rules"][f"up to {m} numbers, score >= {thr}"] = rule_row(pts, lambda pt, m=m, thr=thr: [r["number"] for r in pt["sure"] if r["p_lead"] >= thr][:m])
    R["rules"]["every sure-read rider"] = rule_row(pts, lambda pt: [r["number"] for r in pt["sure"]])
    res[part] = R
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1))

for part in PARTS:
    if part not in res:
        continue
    R = res[part]
    print(f"\n== {part}: {R['points']} points, field {R['mean field']}, riders found {R['riders found per picture']}, with a sure number {R['riders with a sure number per picture']}")
    print("   placed k-th, number read:", " ".join(f"{k}:{v}" for k, v in R["placed k-th: number read"].items()))
    print("   sure-read riders by place:", R["sure-read riders by place"], "| chance", R["chance, a random runner"])
    print(f"   {'rule':58s} {'n/pt':>5s} {'any':>5s} | first 3: A  B  wrong | first 5: A  B  wrong named | first 5 within 3 m: A wrong")
    for name, r in R["rules"].items():
        a, b, c = r["first 3, strict"], r["first 5, strict"], r["first 5, within 3 m"]
        print(f"   {name:58s} {r['numbers per point']:5} {r['points with a number']:5} | {a['numbers right (A)']} {a['points with a right number (B)']} {a['points with a wrong number']} | "
              f"{b['numbers right (A)']} {b['points with a right number (B)']} {b['points with a wrong number']} {b['of the first k, named per point']} | {c['numbers right (A)']} {c['points with a wrong number']}")
