"""winner (2026-11-20), exploration: who is in front is decided at the finish itself, the number is read along that rider's track.
usage: w3_front_track.py grid DAY [DAY ...]
       w3_front_track.py rule SPAN TAU GAP LAST OUT.json DAY [DAY ...]

Data: caps2/dense/A/last/DAY.json (the last 37 s, 5 pictures a second, v11 classifier). CPU only.
Rule: link the riders across the pictures of the last SPAN seconds (fn_tracks.link). In the pictures of the last LAST seconds the rider
furthest right is the front rider; the track that is front in most of them is the winner's track. Average the cap probabilities of that
track over the whole window; call the best cap when its average is TAU or more and, at the finish, the front rider is GAP rider heights
or more ahead of the next (median over the LAST pictures). Cap 11 = any number from 11 up: called only with exactly one such runner.
The official winner is used only to score."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_tracks as ft
from w2_front_rider import races


def call(dr, runners, span, tau, gap, last, first=0.0):
    pics = sorted((p for p in dr["pictures"] if first - 0.05 <= p["to_finish"] <= span + 0.05), key=lambda p: -p["to_finish"])
    if not pics:
        return None, {"why": "no pictures"}
    ids = ft.link(pics)
    fronts, gaps = Counter(), []
    for p, tid in zip(pics, ids):
        if p["to_finish"] > first + last + 0.05 or not p["boxes"]:
            continue
        b = np.array(p["boxes"], float)
        cx, h = (b[:, 0] + b[:, 2]) / 2, b[:, 3] - b[:, 1]
        o = np.argsort(-cx)
        fronts[tid[int(o[0])]] += 1
        gaps.append(float((cx[o[0]] - cx[o[1]]) / max(h[o[0]], 1.0)) if len(o) > 1 else 9.0)
    if not fronts:
        return None, {"why": "no rider at the finish"}
    track, seen_front = fronts.most_common(1)[0]
    P = [np.array(p["probs"][tid.index(track)], float) for p, tid in zip(pics, ids) if track in tid]
    avg = np.mean(P, axis=0)
    c = int(np.argmax(avg))
    info = {"cap": c + 1, "p": round(float(avg[c]), 3), "gap": round(float(np.median(gaps)), 2), "track_pictures": len(P), "front_in": seen_front}
    if len(P) < 3:
        return None, {**info, "why": "short track"}
    if avg[c] < tau:
        return None, {**info, "why": "cap not clear"}
    if info["gap"] < gap:
        return None, {**info, "why": "close"}
    if c + 1 == 11:
        high = [n for n in runners if n >= 11]
        return (high[0], info) if len(high) == 1 else (None, {**info, "why": "several runners from 11 up"})
    return c + 1, info


def evaluate(days, *rule):
    k, rows = Counter(), []
    for day in days:
        for race, dr, rank, runners in races(day):
            n, info = call(dr, runners, *rule)
            win = next((x for x, r in rank.items() if r == 1), None)
            kind = "none" if n is None else ("right" if n == win else "wrong")
            k[kind] += 1
            rows.append({"day": day, "race": race, "agree": dr["clock_agrees"], "call": n, "winner": win, "rank_of_call": rank.get(n), "kind": kind, **info})
    return k, rows


def main():
    if sys.argv[1] == "grid":
        days = sys.argv[2:]
        for last in (0.2, 0.6, 1.0):
            for span in (2.0, 4.0, 6.0):
                for gap in (0.0, 0.3, 0.6):
                    line = []
                    for tau in (0.4, 0.5, 0.6, 0.7):
                        k, _ = evaluate(days, span, tau, gap, last)
                        line.append(f"tau {tau}: {k['right']}+{k['wrong']}w")
                    print(f"last {last} span {span} gap {gap} | " + " | ".join(line))
    else:
        span, tau, gap, last, out, days = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6], sys.argv[7:]
        k, rows = evaluate(days, span, tau, gap, last)
        for r in rows:
            if r["kind"] != "right":
                print(r)
        print(dict(k))
        Path(out).write_text(json.dumps({"rule": {"span": span, "tau": tau, "gap": gap, "last": last}, "days": days, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
