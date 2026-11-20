"""top5 (exploration, 2026-11-20): can a change of place be noticed? ("the horse seen fifth a few seconds ago is now third")
usage: y2_change_look.py OUT.json DAY [DAY ...]          standard library only, no GPU

Same stored data as y1_top5_look.py (bench2/whole/v11, every 3 s; official order from the passing times). A change is looked at
between a time point and the one 6 s earlier. Four things:
 1. how often the official order really changes in 6 s (a horse from 4th or lower into the first three; a new leader);
 2. the identifier's leader said at both times and different: did the new one really pass the old one?
 3. a number said now ("in the first three") and not said 3 s and 6 s earlier: was it really lower before?
 4. a place 1..5 given to the five riders with the highest "in the first three" score in one picture: how often is the place right,
    and when that place goes from 4-5 to 1-3, did the horse really move up?
A look on days already seen; nothing is run again."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc

out_path, days = Path(sys.argv[1]), sys.argv[2:]
GAP = 6
pc = lambda a, b: None if not b else round(100 * a / b, 1)


def place(order, n):
    """Official place of cap number n (11 stands for every cap 11 and above: the best-placed one). None = no such horse."""
    if n >= 11:
        high = [i for i, m in enumerate(order) if m >= 11]
        return high[0] + 1 if high else None
    return order.index(n) + 1 if n in order else None


def horse(order, n):
    q = place(order, n)
    return None if q is None else order[q - 1]


C = Counter()
est_place = Counter()          # (estimated place, official place)
moves = []                     # estimated 4-5 -> 1-3: (official place before, official place now)
for day in days:
    off = fc.official(day)
    whole = json.loads((ROOT / "bench2" / "whole" / "v11" / f"{day}.json").read_text())["races"]
    for race, rr in whole.items():
        pts = {}
        for p in rr["points"]:
            ref = fc.reference_full(off[race], p["race_time"])
            if ref is None or len(ref["order"]) < 6:
                continue
            best = {}
            for r in p.get("riders_now", []):
                if r.get("sure") and r["number"] and (r["number"] not in best or r["p_lead"] > best[r["number"]]):
                    best[r["number"]] = r["p_lead"]
            est = sorted(best, key=lambda n: -best[n])[:5]
            pts[p["to_finish"]] = {"p": p, "order": ref["order"], "d": ref["to_go"], "est": est}
            for i, n in enumerate(est):
                q = place(ref["order"], n)
                est_place[(i + 1, q if q is not None and q <= 6 else "7+" if q else "none")] += 1
        for tau, now in pts.items():
            old, mid = pts.get(tau + GAP), pts.get(tau + GAP // 2)
            if old is None:
                continue
            C["pairs"] += 1
            o0, o1, d0, d1 = old["order"], now["order"], old["d"], now["d"]
            # 1. what really happens in 6 s
            risen = [n for n in o1[:3] if n in o0 and o0.index(n) >= 3]
            C["pairs: a horse came into the first three"] += bool(risen)
            C["pairs: a new leader"] += o1[0] != o0[0]
            C["pairs: a new leader, clear (3 m in front now, 3 m behind before)"] += (o1[0] != o0[0] and o1[0] in d0 and d0[o1[0]] - d0[o0[0]] >= fc.MARGIN_M
                                                                                    and d1[o1[1]] - d1[o1[0]] >= fc.MARGIN_M)
            # 2. the leader said at both times
            a, b = old["p"].get("leader"), now["p"].get("leader")
            if a and b:
                C["leader said at both times"] += 1
                real = o1[0] != o0[0]
                C["leader said at both times: really a new leader"] += real
                if a != b:
                    C["leader changes said"] += 1
                    ha0, hb0, ha1, hb1 = horse(o0, a), horse(o0, b), horse(o1, a), horse(o1, b)
                    passed = None not in (ha0, hb0, ha1, hb1) and d0[ha0] < d0[hb0] and d1[hb1] < d1[ha1]
                    C["leader changes said: new one is the leader now"] += hb1 == o1[0]
                    C["leader changes said: new one really passed the old one"] += passed
                    C["leader changes said: both right (old led then, new leads now)"] += ha0 == o0[0] and hb1 == o1[0]
                    C["leader changes said: no pass, the new one was already in front"] += None not in (ha0, hb0) and d0[hb0] <= d0[ha0]
                if real:
                    C["real new leader, caught (said the new leader, and another one before)"] += a != b and horse(o1, b) == o1[0]
            # 3. a number said now and not before
            s1, s0 = set(now["p"].get("statements", [])), set(old["p"].get("statements", []))
            sm = set(mid["p"].get("statements", [])) if mid else set()
            if s0 and mid is not None and sm:
                for n in s1 - s0 - sm:
                    q0, q1 = place(o0, n), place(o1, n)
                    C["new numbers said"] += 1
                    if q1 is None or q1 > 3:
                        C["new numbers said: not in the first three now"] += 1
                    elif q0 is not None and q0 <= 3:
                        C["new numbers said: in the first three, and was there 6 s ago too"] += 1
                    else:
                        C["new numbers said: really came into the first three"] += 1
                # the real arrivals: were they said as new?
                for n in risen:
                    C["real arrivals in the first three (with numbers said before)"] += 1
                    C["real arrivals: said now and not before"] += any(horse(o1, m) == n for m in s1 - s0 - sm)
                    C["real arrivals: said now"] += any(horse(o1, m) == n for m in s1)
            # 4. estimated place 4-5 -> 1-3
            for n in now["est"][:3]:
                if n in old["est"][3:5]:
                    moves.append((place(o0, n), place(o1, n)))
            # 5. the same with a steadier rule: the same thing said at two time points in a row before (9 s and 6 s ago) and after (3 s ago, now)
            far = pts.get(tau + 9)
            if far is not None and mid is not None:
                of, df = far["order"], far["d"]
                L = [x["p"].get("leader") for x in (far, old, mid, now)]
                if all(L) and L[0] == L[1] and L[2] == L[3] and L[0] != L[2]:
                    ha0, hb0, ha1, hb1 = horse(of, L[0]), horse(of, L[2]), horse(o1, L[0]), horse(o1, L[2])
                    C["steady leader changes said"] += 1
                    C["steady leader changes said: new one is the leader now"] += hb1 == o1[0]
                    C["steady leader changes said: new one really passed the old one"] += None not in (ha0, hb0, ha1, hb1) and df[ha0] < df[hb0] and d1[hb1] < d1[ha1]
                S = [set(x["p"].get("statements", [])) for x in (far, old, mid, now)]
                if S[0] and S[1]:
                    for n in (S[2] & S[3]) - S[0] - S[1]:
                        q0, q1 = place(of, n), place(o1, n)
                        C["steady new numbers said"] += 1
                        C["steady new numbers said: really came into the first three"] += q1 is not None and q1 <= 3 and (q0 is None or q0 > 3)
                        C["steady new numbers said: was in the first three 9 s ago too"] += q1 is not None and q1 <= 3 and q0 is not None and q0 <= 3
                        C["steady new numbers said: not in the first three now"] += q1 is None or q1 > 3

res = {"days": days, "gap_seconds": GAP, "counts": dict(C)}
tot = Counter()
for (e, q), v in est_place.items():
    tot[e] += v
res["place by score, one picture"] = {
    str(e): {"riders": tot[e], "official place the same": pc(est_place[(e, e)], tot[e]),
             "within one place": pc(sum(est_place[(e, q)] for q in (e - 1, e, e + 1)), tot[e]),
             "in the first five": pc(sum(est_place[(e, q)] for q in range(1, 6)), tot[e]),
             "official places": {str(q): est_place[(e, q)] for q in (1, 2, 3, 4, 5, 6, "7+", "none")}} for e in range(1, 6)}
ok = [(a, b) for a, b in moves if a is not None and b is not None]
res["place by score went from 4-5 to 1-3"] = {
    "events": len(moves), "really moved up (any number of places)": pc(sum(b < a for a, b in ok), len(moves)),
    "really from 4th or lower into the first three": pc(sum(a >= 4 and b <= 3 for a, b in ok), len(moves)),
    "same official place at both times": pc(sum(a == b for a, b in ok), len(moves)),
    "really moved back": pc(sum(b > a for a, b in ok), len(moves)),
    "was in the first three at both times": pc(sum(a <= 3 and b <= 3 for a, b in ok), len(moves))}
out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1))

print(f"{len(days)} days, pairs of time points {GAP} s apart: {C['pairs']}")
for k in sorted(C):
    if k != "pairs":
        print(f"  {k}: {C[k]}")
print("place by score (one picture):")
for e, r in res["place by score, one picture"].items():
    print(f"  given place {e}: {r['riders']} riders; same place {r['official place the same']}%, within one {r['within one place']}%, in the first five {r['in the first five']}% | {r['official places']}")
print("place by score 4-5 -> 1-3:", res["place by score went from 4-5 to 1-3"])
