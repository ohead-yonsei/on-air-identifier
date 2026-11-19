import json, sys
from collections import Counter
sys.path.insert(0, "code")
import w2_front_rider as w2
old = json.load(open("winner/explore/look_A.json"))
groups = {"dev": "20261115 otherday otherday2 otherday3 otherday4 otherday5".split()}
if len(sys.argv) > 1:
    groups["held"] = "otherday9 otherday10 otherday11 otherday14 otherday15 otherday16 otherday17 otherday18 otherday19".split()
for gname, days in groups.items():
    print("=====", gname)
    for span, tau in ((4.0, 0.5), (3.0, 0.5), (4.0, 0.4), (6.0, 0.5), (6.0, 0.4), (4.0, 0.6)):
        m, combos = Counter(), {k: Counter() for k in ("old", "front", "agree only", "old unless front differs, plus front alone", "front first, else old", "old, plus front alone")}
        for day in days:
            for race, dr, rank, runners in w2.races(day):
                win = next(x for x, r in rank.items() if r == 1)
                o = old[f"{day}/{race}"]["call"]
                f, info = w2.call(dr, runners, span, tau, 0.0)
                k = lambda c: "none" if c is None else ("right" if c == win else "wrong")
                m[(k(o), k(f))] += 1
                combos["old"][k(o)] += 1
                combos["front"][k(f)] += 1
                combos["agree only"][k(o if o == f else None)] += 1
                combos["old unless front differs, plus front alone"][k(None if (o is not None and f is not None and o != f) else (o if o is not None else f))] += 1
                combos["front first, else old"][k(f if f is not None else o)] += 1
                combos["old, plus front alone"][k(o if o is not None else f)] += 1
        print(f"front rule span {span} tau {tau}: (old, front) ->", dict(sorted(m.items())))
        for name, c in combos.items():
            n = sum(c.values())
            print(f"    {name:45s} called {c['right'] + c['wrong']}/{n} right {c['right']} wrong {c['wrong']} ({100 * c['right'] / max(c['right'] + c['wrong'], 1):.0f}% of calls, {100 * c['right'] / n:.0f}% of races)")
