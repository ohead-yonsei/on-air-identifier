"""far (2026-11-20), exploration: votes from the rider that is in front by position, instead of the strip-trained leader model.
usage: w8_geo_vote.py grid GROUP=DAY,DAY,... [GROUP=...]

Per picture of the last 6 s the "front rider" is the one furthest right (the home straight in Seoul runs left to right on screen).
  geo     it votes for its number when its cap probability is SURE or more
  model   the current rule: the strip-trained leader model's pick among the sure riders
  either  a picture votes for the geo number and for the model number (once each)
The candidate is the number with the most votes, NEED or more; then the condition of the winner experiment (still in front in the
last 2 s). Cap 11 = every number from 11 up: a candidate only when the race has exactly one such runner.
The official winner is used only to score. Exploration on days already seen."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft
import w4_winner_guard as w4


def votes(pics, runners, sure=0.7):
    last = [p for p in pics if -0.05 <= p["to_finish"] <= w4.SPAN + 0.05]
    who = ft.identify(last, ft.link(last), memory=False)
    geo, model = [], []
    high = [n for n in runners if n >= 11]
    for p, w in zip(last, who):
        g = None
        if p["boxes"]:
            b, P = np.array(p["boxes"], float), np.array(p["probs"], float)
            i = int(np.argmax((b[:, 0] + b[:, 2]) / 2))
            c = int(np.argmax(P[i]))
            if P[i, c] >= sure:
                g = c + 1 if c < 10 else (high[0] if len(high) == 1 else None)
        geo.append(g)
        model.append(ft.decide(p, w, w4.ORDER, w4.STRIP)["pick"] if p["boxes"] else None)
    return geo, model


def decide(pics, runners, kind, need, sure=0.7, guard=True):
    geo, model = votes(pics, runners, sure)
    c = Counter()
    for g, m in zip(geo, model):
        for n in {"geo": {g}, "model": {m}, "either": {g, m}}[kind]:
            if n is not None:
                c[n] += 1
    top = c.most_common(2)
    if not top or top[0][1] < need:
        return None, None
    cand = top[0][0]
    if not guard:
        return cand, cand
    seen, front = w4.in_front(pics, cand, w4.GUARD_T)
    return cand, (cand if seen >= w4.GUARD_SEEN and front >= w4.GUARD_SHARE * seen else None)


def main():
    groups = dict(a.split("=") for a in sys.argv[2:])
    data = {}
    for g, days in groups.items():
        data[g] = []
        for day in days.split(","):
            fc.FOLDER.setdefault(day, day)
            off = fc.official(day)
            for race, dr in sorted(w4.dense(day).items()):
                rank = {h["number"]: h["rank"] for h in off[race]["horses"] if h.get("rank")}
                data[g].append((dr["pictures"], sorted(h["number"] for h in off[race]["horses"]), next((x for x, r in rank.items() if r == 1), None)))
    for kind in ("model", "geo", "either"):
        for need in (5, 7, 9):
            for guard in (True, False):
                line = []
                for g, rows in data.items():
                    k = Counter()
                    for pics, runners, win in rows:
                        _, n = decide(pics, runners, kind, need, guard=guard)
                        k["none" if n is None else ("right" if n == win else "wrong")] += 1
                    line.append(f"{g} {len(rows)}: {k['right']}+{k['wrong']}w")
                print(f"{kind:6s} need {need} {'with the condition' if guard else 'no condition      '} | " + " | ".join(line))


if __name__ == "__main__":
    main()
