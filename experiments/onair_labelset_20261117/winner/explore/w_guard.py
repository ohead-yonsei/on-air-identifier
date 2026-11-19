import json, sys
from collections import Counter
sys.path.insert(0, "code")
import numpy as np
import w2_front_rider as w2
old = json.load(open("winner/explore/look_A.json"))
days = sys.argv[1:]
rows = []
for d in days:
    for r, dr, rank, run in w2.races(d):
        o = old[d + "/" + r]
        n = o["call"]
        if n is None: continue
        win = o["winner"]; cap = min(n, 11) - 1
        res = {}
        for T in (1.0, 2.0, 3.0):
            seen = front = ahead_any = 0
            for p in dr["pictures"]:
                if p["to_finish"] > T + 0.05 or not p["boxes"]: continue
                b, P = np.array(p["boxes"], float), np.array(p["probs"], float)
                cx, h = (b[:, 0] + b[:, 2]) / 2, b[:, 3] - b[:, 1]
                i = int(np.argmax(P[:, cap]))
                if P[i, cap] < 0.5: continue
                seen += 1
                a = int(((cx - cx[i]) > 0.3 * h[i]).sum())      # riders clearly ahead of it
                front += a == 0
            res[T] = (seen, front)
        rows.append((n == win, d, r, n, win, rank.get(n), o["counts"], res))
for ok in (False, True):
    print("=== calls that were", "right" if ok else "wrong")
    for row in rows:
        if row[0] == ok:
            print(row[1], row[2], "call", row[3], "win", row[4], "rank", row[5], "| last 1 s seen/front %d/%d | 2 s %d/%d | 3 s %d/%d" % (*row[7][1.0], *row[7][2.0], *row[7][3.0]), row[6])
for T in (1.0, 2.0, 3.0):
    for need_seen in (1, 2, 3):
        for q in (0.5, 0.67, 0.8, 1.0):
            c = Counter()
            for ok, d, r, n, win, rk, cnt, res in rows:
                s, f = res[T]
                keep = s >= need_seen and f >= q * s
                c[("right" if ok else "wrong") + (" kept" if keep else " dropped")] += 1
            print(f"T {T} seen>={need_seen} front share>={q}: kept right {c['right kept']} wrong {c['wrong kept']} | dropped right {c['right dropped']} wrong {c['wrong dropped']}")
