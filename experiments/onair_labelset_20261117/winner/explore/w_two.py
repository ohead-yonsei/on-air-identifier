import json, sys
from collections import Counter
sys.path.insert(0, "code")
import numpy as np
import w2_front_rider as w2
old = json.load(open("winner/explore/look_A.json"))
days = sys.argv[1:]
data = [(day, race, dr, rank, runners) for day in days for race, dr, rank, runners in w2.races(day)]
def number(c, runners):
    if c == 11:
        high = [x for x in runners if x >= 11]
        return high[0] if len(high) == 1 else None
    return c
print(len(data), "races")
for span in (5.0, 6.0, 7.0):
    for hi in (0.45, 0.5):
        for lo in (0.2, 0.25, 0.3, 0.35):
            c = Counter()
            for day, race, dr, rank, runners in data:
                win = next(x for x, r in rank.items() if r == 1)
                fr = [f for f in (w2.front(p) for p in dr["pictures"] if -0.05 <= p["to_finish"] <= span + 0.05) if f is not None]
                o = old[f"{day}/{race}"]["call"]
                n = None
                if len(fr) >= 3:
                    avg = np.mean([f[0] for f in fr], axis=0); k = int(np.argmax(avg)); top = number(k + 1, runners)
                    if avg[k] >= hi: n = top
                    elif o is not None and top == o and avg[k] >= lo: n = o
                c["none" if n is None else ("right" if n == win else "wrong")] += 1
            print(f"span {span} hi {hi} lo {lo}: right {c['right']} wrong {c['wrong']} none {c['none']}  ({100 * c['right'] / max(c['right'] + c['wrong'], 1):.0f}% of calls, {100 * c['right'] / len(data):.0f}% of races)")
