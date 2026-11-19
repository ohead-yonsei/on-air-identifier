import sys
from collections import Counter
sys.path.insert(0, "code")
import numpy as np
import fn_common as fc, w4_winner_guard as w4, w8_geo_vote as w8
days = sys.argv[1:]
agree = Counter()
for day in days:
    fc.FOLDER.setdefault(day, day)
    off = fc.official(day)
    for race, dr in sorted(w4.dense(day).items()):
        rank = {h["number"]: h["rank"] for h in off[race]["horses"] if h.get("rank")}
        runners = sorted(h["number"] for h in off[race]["horses"])
        win = next(x for x, r in rank.items() if r == 1)
        pics = dr["pictures"]
        geo, model = w8.votes(pics, runners)
        cg, cm = Counter(g for g in geo if g is not None), Counter(m for m in model if m is not None)
        _, ng = w8.decide(pics, runners, "geo", 7)
        _, nm = w8.decide(pics, runners, "model", 5)
        k = lambda n: "none" if n is None else ("right" if n == win else "wrong")
        agree[(k(nm), k(ng))] += 1
        if ng is not None and ng != win or (nm is None and ng is not None):
            tfs = [p["to_finish"] for p in pics if -0.05 <= p["to_finish"] <= 6.05]
            line = "".join(str(g % 10) if g is not None else "-" for g in geo)
            seen, front = w4.in_front(pics, ng, 2.0)
            top2 = cg.most_common(2)
            print(f"{day:10s} {race} {'  ' if dr['clock_agrees'] else 'nc'} win {win:2d} geo {ng:2d} ({k(ng)}, rank {rank.get(ng)}) model {nm} | geo votes {dict(cg.most_common(3))} model votes {dict(cm.most_common(3))} | last2s seen/front {seen}/{front} | {line}")
print(dict(agree))
