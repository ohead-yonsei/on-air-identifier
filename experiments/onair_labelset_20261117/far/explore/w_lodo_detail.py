import sys
from collections import Counter
sys.path.insert(0, "code")
import w9_straight_model as w9, w4_winner_guard as w4
data = {d: w9.rows(d, 6.0) for d in w9.D15}
tot = {k: Counter() for k in ("need5", "need7", "need5 ratio2", "need7 ratio1.5", "need5, strip must not disagree", "need7, strip must not disagree")}
for d in w9.D15:
    X = [x for o in w9.D15 if o != d for x in data[o][0]]; y = [v for o in w9.D15 if o != d for v in data[o][1]]
    m = w9.fit(X, y)
    for race, dr, off, win in w9.day_races(d):
        rank = {h["number"]: h["rank"] for h in off["horses"] if h.get("rank")}
        cand, n5, votes = w9.call(dr["pictures"], m)
        _, n7, _ = w9.call(dr["pictures"], m, need=7)
        strip_cand = w4.call(dr["pictures"])[1]["candidate"]
        top = Counter(votes).most_common(2)
        second = top[1][1] if len(top) > 1 else 0
        k = lambda n: "none" if n is None else ("right" if n == win else "wrong")
        tot["need5"][k(n5)] += 1; tot["need7"][k(n7)] += 1
        tot["need5 ratio2"][k(n5 if n5 is not None and top[0][1] >= 2 * second else None)] += 1
        tot["need7 ratio1.5"][k(n7 if n7 is not None and top[0][1] >= 1.5 * second else None)] += 1
        tot["need5, strip must not disagree"][k(n5 if strip_cand in (None, n5) else None)] += 1
        tot["need7, strip must not disagree"][k(n7 if strip_cand in (None, n7) else None)] += 1
        if n5 is not None and n5 != win:
            seen, front = w4.in_front(dr["pictures"], n5, 2.0)
            print(d, race, "agree" if dr["clock_agrees"] else "nc", "win", win, "call", n5, "rank", rank.get(n5), "votes", dict(Counter(votes).most_common(3)), "need7 call", n7, "strip cand", strip_cand, "seen/front", seen, front)
for name, c in tot.items():
    print(f"{name:34s} right {c['right']} wrong {c['wrong']} none {c['none']}")
