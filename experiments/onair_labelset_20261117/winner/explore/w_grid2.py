import sys
from collections import Counter
sys.path.insert(0, "code")
import numpy as np
import w2_front_rider as w2
days = "20261115 otherday otherday2 otherday3 otherday4 otherday5".split()
data = [(day, race, dr, rank, runners) for day in days for race, dr, rank, runners in w2.races(day)]
print(len(data), "races")
for span in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0):
    line = []
    for tau in (0.3, 0.35, 0.4, 0.45, 0.5):
        c = Counter()
        for day, race, dr, rank, runners in data:
            win = next(x for x, r in rank.items() if r == 1)
            f, info = w2.call(dr, runners, span, tau, 0.0)
            c["none" if f is None else ("right" if f == win else "wrong")] += 1
        line.append(f"tau {tau}: {c['right']}+{c['wrong']}w")
    print(f"span {span:4.1f} | " + " | ".join(line))
# margin between the best and the second cap of the averaged front rider
print("with a margin (best minus second average), span 6:")
for tau in (0.3, 0.4):
    for margin in (0.0, 0.1, 0.2, 0.3):
        c = Counter()
        for day, race, dr, rank, runners in data:
            win = next(x for x, r in rank.items() if r == 1)
            fr = [f for f in (w2.front(p) for p in dr["pictures"] if -0.05 <= p["to_finish"] <= 6.05) if f is not None]
            if len(fr) < 3: c["none"] += 1; continue
            avg = np.mean([f[0] for f in fr], axis=0); o = np.argsort(-avg)
            n = int(o[0]) + 1
            if avg[o[0]] < tau or avg[o[0]] - avg[o[1]] < margin: c["none"] += 1; continue
            if n == 11:
                high = [x for x in runners if x >= 11]
                if len(high) != 1: c["none"] += 1; continue
                n = high[0]
            c["right" if n == win else "wrong"] += 1
        print(f"  tau {tau} margin {margin}: right {c['right']} wrong {c['wrong']} none {c['none']}")
