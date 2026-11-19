import sys
from collections import Counter
sys.path.insert(0, "code")
import numpy as np
import fn_common as fc, fn_tracks as ft, w4_winner_guard as w4
G = {"nine": "otherday9 otherday10 otherday11 otherday14 otherday15 otherday16 otherday17 otherday18 otherday19".split(), "three": "otherday20 otherday21 otherday22".split(),
     "6,7,8": "otherday6 otherday7 otherday8".split(), "trained six": "20261115 otherday otherday2 otherday3 otherday4 otherday5".split()}
def vote_mem(pics, window, need=5):
    last = sorted((p for p in pics if -0.05 <= p["to_finish"] <= 8.05), key=lambda p: -p["to_finish"])
    who = ft.identify(last, ft.link(last), memory=True, window=window)
    c = Counter()
    for p, w in zip(last, who):
        if p["to_finish"] <= w4.SPAN + 0.05:
            n = ft.decide(p, w, w4.ORDER, w4.STRIP)["pick"]
            if n is not None: c[n] += 1
    top = c.most_common(1)
    return top[0][0] if top and top[0][1] >= need else None
for g, days in G.items():
    tot = {k: Counter() for k in ("current", "memory 5", "memory 10", "memory 25")}
    for d in days:
        fc.FOLDER.setdefault(d, d); off = fc.official(d)
        for race, dr in sorted(w4.dense(d).items()):
            rank = {h["number"]: h["rank"] for h in off[race]["horses"] if h.get("rank")}
            win = next(x for x, r in rank.items() if r == 1)
            k = lambda n: "none" if n is None else ("right" if n == win else "wrong")
            tot["current"][k(w4.call(dr["pictures"])[0])] += 1
            for w in (5, 10, 25):
                cand = vote_mem(dr["pictures"], w)
                n = None
                if cand is not None:
                    seen, front = w4.in_front(dr["pictures"], cand, w4.GUARD_T)
                    n = cand if seen >= 1 and front >= 0.5 * seen else None
                tot[f"memory {w}"][k(n)] += 1
    print(g, " | ".join(f"{name}: {c['right']}+{c['wrong']}w" for name, c in tot.items()))
