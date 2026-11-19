"""winner (2026-11-20), exploration: call the winner from the FRONT rider of the home-straight pictures.
usage: w2_front_rider.py look DAY [DAY ...]            where the winner's cap sits among the riders in the last seconds (left/right)
       w2_front_rider.py grid DAY [DAY ...]            a small grid of rules (prints called / right / wrong)
       w2_front_rider.py rule SPAN TAU GAP OUT.json DAY [DAY ...]    one rule, per race

Data: caps2/dense/A/last/DAY.json (riders of the last 37 s, 5 pictures a second, cap probabilities of the v11 classifier). CPU only.
Rule: in the pictures of the last SPAN seconds, take the rider furthest to the right (the home straight in Seoul is filmed from the
stands: the field runs left to right). Average that rider's cap probabilities over the pictures; call the cap with the highest average
when it is TAU or more and the front rider is, in the middle picture of the window (median), at least GAP rider heights ahead of the next.
Cap 11 stands for every number from 11 up: called only when the race has exactly one such runner.
The answer (official winner) is used only to score."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc


def races(day, name="A"):
    fc.FOLDER.setdefault(day, day)
    off = fc.official(day)
    for race, dr in sorted(json.loads((ROOT / "caps2" / "dense" / name / "last" / f"{day}.json").read_text())["races"].items()):
        rank = {h["number"]: h["rank"] for h in off[race]["horses"] if h.get("rank")}
        runners = sorted(h["number"] for h in off[race]["horses"])
        yield race, dr, rank, runners


def front(p):
    """(probabilities of the rider furthest right, gap to the next rider in rider heights, riders) or None"""
    if not p["boxes"]:
        return None
    b = np.array(p["boxes"], float)
    cx, h = (b[:, 0] + b[:, 2]) / 2, b[:, 3] - b[:, 1]
    o = np.argsort(-cx)
    gap = float((cx[o[0]] - cx[o[1]]) / max(h[o[0]], 1.0)) if len(o) > 1 else 9.0
    return np.array(p["probs"][int(o[0])], float), gap, len(o)


def call(dr, runners, span, tau, gap, first=0.0):
    fr = [f for f in (front(p) for p in dr["pictures"] if first - 0.05 <= p["to_finish"] <= span + 0.05) if f is not None]
    if len(fr) < 3:
        return None, {"why": "few pictures"}
    avg = np.mean([f[0] for f in fr], axis=0)
    c = int(np.argmax(avg))
    info = {"cap": c + 1, "p": round(float(avg[c]), 3), "gap": round(float(np.median([f[1] for f in fr])), 2), "pictures": len(fr)}
    if avg[c] < tau:
        return None, {**info, "why": "cap not clear"}
    if info["gap"] < gap:
        return None, {**info, "why": "close"}
    if c + 1 == 11:
        high = [n for n in runners if n >= 11]
        return (high[0], info) if len(high) == 1 else (None, {**info, "why": "several runners from 11 up"})
    return c + 1, info


def evaluate(days, span, tau, gap, first=0.0):
    k, rows = Counter(), []
    for day in days:
        for race, dr, rank, runners in races(day):
            n, info = call(dr, runners, span, tau, gap, first)
            win = next((x for x, r in rank.items() if r == 1), None)
            kind = "none" if n is None else ("right" if n == win else "wrong")
            k[kind] += 1
            k[("agree" if dr["clock_agrees"] else "nc", kind)] += 1
            rows.append({"day": day, "race": race, "agree": dr["clock_agrees"], "call": n, "winner": win, "rank_of_call": rank.get(n), "kind": kind, **info})
    return k, rows


def main():
    mode = sys.argv[1]
    if mode == "look":
        tot = Counter()
        for day in sys.argv[2:]:
            for race, dr, rank, runners in races(day):
                wcap = min(next(x for x, r in rank.items() if r == 1), 11) - 1
                for p in dr["pictures"]:
                    if p["to_finish"] > 3.05 or not p["boxes"]:
                        continue
                    b, P = np.array(p["boxes"], float), np.array(p["probs"], float)
                    cx = (b[:, 0] + b[:, 2]) / 2
                    i = int(np.argmax(P[:, wcap]))
                    if P[i, wcap] < 0.7:
                        tot["winner's cap not seen (0.7)"] += 1
                        continue
                    r = int((cx > cx[i]).sum())
                    tot["winner's cap is the rightmost" if r == 0 else ("second from the right" if r == 1 else ("leftmost" if (cx < cx[i]).sum() == 0 else "elsewhere"))] += 1
        n = sum(tot.values())
        print({k: f"{v} ({100 * v / n:.1f}%)" for k, v in tot.most_common()})
    elif mode == "grid":
        days = sys.argv[2:]
        for first in (0.0, 0.4):
            for span in (1.0, 2.0, 3.0, 4.0):
                for tau in (0.5, 0.6, 0.7, 0.8):
                    for gap in (0.0, 0.5, 1.0):
                        k, _ = evaluate(days, span, tau, gap, first)
                        n = k["right"] + k["wrong"] + k["none"]
                        print(f"from {first} span {span} tau {tau} gap {gap}: called {k['right'] + k['wrong']}/{n}, right {k['right']}, wrong {k['wrong']}  ({100 * k['right'] / max(k['right'] + k['wrong'], 1):.1f}% of calls)")
    else:
        span, tau, gap, out, days = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6:]
        k, rows = evaluate(days, span, tau, gap)
        for r in rows:
            if r["kind"] != "right":
                print(r)
        print({a: b for a, b in k.items() if isinstance(a, str)}, {f"{a} {b}": v for (a, b), v in ((x, y) for x, y in k.items() if isinstance(x, tuple))})
        Path(out).write_text(json.dumps({"rule": {"span": span, "tau": tau, "gap": gap}, "days": days, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
