"""far (2026-11-20), exploration: why is there no winner candidate in a third of the races?
usage: w7_far_look.py DAY [DAY ...]        (CPU; the dense pictures of the last seconds, v11 classifier: winner/dense or caps2/dense/A/last)

For every race, over the 31 pictures of the last 6 s:
  cap seen      pictures in which some rider has the winner's cap at 0.7 or more (the bar for a "sure" rider)
  seen, front   ... and no other rider is further right (ahead)
  picked        pictures in which the strip-trained leader model picks the winner (0.5 or more)  = the votes of the current rule
  riders, size  riders per picture and their median height in pixels
Races are grouped by what the current rule does (right call, wrong call, no candidate, candidate dropped by the condition).
The official winner is used only to describe; nothing here is a test."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft
import w4_winner_guard as w4


def race_stats(pics, win):
    cap = min(win, 11) - 1
    last = [p for p in pics if -0.05 <= p["to_finish"] <= w4.SPAN + 0.05]
    who = ft.identify(last, ft.link(last), memory=False)
    seen = front = picked = 0
    hs, ns, best = [], [], []
    for p, w in zip(last, who):
        ns.append(len(p["boxes"]))
        if not p["boxes"]:
            continue
        b, P = np.array(p["boxes"], float), np.array(p["probs"], float)
        cx, h = (b[:, 0] + b[:, 2]) / 2, b[:, 3] - b[:, 1]
        hs.append(float(np.median(h)))
        i = int(np.argmax(P[:, cap]))
        best.append(float(P[i, cap]))
        if P[i, cap] >= 0.7:
            seen += 1
            front += int(((cx - cx[i]) > w4.AHEAD * h[i]).sum()) == 0
        picked += ft.decide(p, w, w4.ORDER, w4.STRIP)["pick"] == win
    return {"pictures": len(last), "seen": seen, "front": front, "picked": picked, "riders": float(np.mean(ns)) if ns else 0.0,
            "height": float(np.median(hs)) if hs else 0.0, "best": float(np.median(best)) if best else 0.0}


def main():
    groups, lines = defaultdict(list), []
    for day in sys.argv[1:]:
        fc.FOLDER.setdefault(day, day)
        off = fc.official(day)
        for race, dr in sorted(w4.dense(day).items()):
            rank = {h["number"]: h["rank"] for h in off[race]["horses"] if h.get("rank")}
            win = next((x for x, r in rank.items() if r == 1), None)
            n, info = w4.call(dr["pictures"])
            cand = info["candidate"]
            kind = "no candidate" if cand is None else ("dropped by the condition" if n is None else ("right" if n == win else "wrong"))
            s = race_stats(dr["pictures"], win)
            groups[kind].append(s)
            if kind in ("no candidate", "dropped by the condition"):
                lines.append(f"  {day:10s} {race} win {win:2d} {kind:24s} seen {s['seen']:2d} front {s['front']:2d} picked {s['picked']:2d} | riders {s['riders']:4.1f} height {s['height']:5.1f} best cap prob {s['best']:.2f} | votes {info['pictures_naming_each']}")
    for kind, ss in groups.items():
        f = lambda k: np.mean([s[k] for s in ss])
        print(f"{kind:26s} races {len(ss):3d}: cap seen in {f('seen'):4.1f} of 31 pictures, seen and in front {f('front'):4.1f}, picked {f('picked'):4.1f} | riders {f('riders'):4.1f}, height {f('height'):5.1f} px, median best cap prob {f('best'):.2f}")
    nc = groups["no candidate"]
    print("no candidate:", len(nc), "races;", sum(s["seen"] >= 10 for s in nc), "with the winner's cap seen in 10+ pictures;", sum(s["front"] >= 5 for s in nc), "seen and in front in 5+;", sum(s["seen"] < 5 for s in nc), "seen in fewer than 5")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
