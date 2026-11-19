import json, sys
from collections import Counter
sys.path.insert(0, "code")
import w2_front_rider as w2
old = json.load(open("winner/explore/look_A.json"))
G = {"dev (6 days)": "20261115 otherday otherday2 otherday3 otherday4 otherday5".split(), "held (9 days)": "otherday9 otherday10 otherday11 otherday14 otherday15 otherday16 otherday17 otherday18 otherday19".split(),
     "held, days new for the classifier (16-19)": "otherday16 otherday17 otherday18 otherday19".split()}
for g, days in G.items():
    data = [(d, r, dr, rank, run) for d in days for r, dr, rank, run in w2.races(d)]
    for name, fn in (("old vote", lambda d, r, dr, run: old[d + "/" + r]["call"]), ("front 6.0/0.40", lambda d, r, dr, run: w2.call(dr, run, 6.0, 0.40, 0.0)[0]), ("front 6.0/0.45", lambda d, r, dr, run: w2.call(dr, run, 6.0, 0.45, 0.0)[0])):
        c, ca = Counter(), Counter()
        for d, r, dr, rank, run in data:
            win = next(x for x, k in rank.items() if k == 1); n = fn(d, r, dr, run)
            k = "none" if n is None else ("right" if n == win else "wrong")
            c[k] += 1
            if dr["clock_agrees"]: ca[k] += 1
        print("%-42s %-15s all %d: right %d wrong %d none %d | clock agrees %d: right %d wrong %d none %d" % (g, name, sum(c.values()), c["right"], c["wrong"], c["none"], sum(ca.values()), ca["right"], ca["wrong"], ca["none"]))
if len(sys.argv) > 1:
    for d in G["held (9 days)"]:
        for r, dr, rank, run in w2.races(d):
            win = next(x for x, k in rank.items() if k == 1); n, info = w2.call(dr, run, 6.0, 0.40, 0.0)
            if n != win: print(d, r, "agree" if dr["clock_agrees"] else "nc", "win", win, "call", n, "old", old[d + "/" + r]["call"], info, "rank of call", rank.get(n))
