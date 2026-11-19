import sys
from collections import Counter
sys.path.insert(0, "code")
import joblib
import w9_straight_model as w9, w4_winner_guard as w4
m = joblib.load("far/explore/straight_6s_all15.joblib")
for group, days in (("three days of the winner test", ["otherday20", "otherday21", "otherday22"]), ("6, 7, 8 (mined-label days)", ["otherday6", "otherday7", "otherday8"])):
    tot = {k: Counter() for k in ("strip (current)", "straight need 5", "straight need 7")}
    for d in days:
        for race, dr, off, win in w9.day_races(d):
            rank = {h["number"]: h["rank"] for h in off["horses"] if h.get("rank")}
            k = lambda n: "none" if n is None else ("right" if n == win else "wrong")
            old = w4.call(dr["pictures"])[0]
            c5, n5, votes = w9.call(dr["pictures"], m); _, n7, _ = w9.call(dr["pictures"], m, need=7)
            tot["strip (current)"][k(old)] += 1; tot["straight need 5"][k(n5)] += 1; tot["straight need 7"][k(n7)] += 1
            if (n5, n7) != (old, old):
                print(" ", d, race, "agree" if dr["clock_agrees"] else "nc", "win", win, "| current", old, "| need5", n5, "need7", n7, "| votes", dict(Counter(votes).most_common(3)), "rank of call", rank.get(n5))
    for name, c in tot.items():
        print(f"{group:32s} {name:18s} right {c['right']} wrong {c['wrong']} none {c['none']}")
