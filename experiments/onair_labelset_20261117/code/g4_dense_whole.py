"""finish3 stage 2: riders of every picture over the WHOLE race, two pictures a second (boxes and cap probabilities, as fn6_dense.py does for
the last 37 s). usage: g4_dense_whole.py DAY [RACE ...]

Pictures: the winner's finish and every 0.5 s before it, back to race time 7 s. The video is decoded once from the first wanted frame.
Writes finish3/dense/DAY.json. Nothing here reads the strip; the official time only places the pictures (training and exploration data)."""
import bisect
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "identifier"))
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
from fn6_dense import riders
from identify_l1 import LeadingGroupIdentifier, video_pts

STEP, FIRST = 0.5, 7.0


def main():
    day, only = sys.argv[1], set(sys.argv[2:])
    fc.FOLDER.setdefault(day, day)
    ident = LeadingGroupIdentifier(fast=True)
    off, tms = fc.official(day), fc.times(day)
    out_path = ROOT / "finish3" / "dense" / f"{day}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "finish3" / "pts" / day).mkdir(parents=True, exist_ok=True)
    res = {"day": day, "step": STEP, "first_race_time": FIRST, "races": {}}
    if out_path.exists():
        res["races"] = json.loads(out_path.read_text())["races"]
    t0 = time.time()
    for race in sorted(off):
        if (only and race not in only) or "error" in off[race] or race in res["races"]:
            continue
        tw = fc.winner_time(off[race])
        vf, how, agree = fc.video_finish(tms.get(race), tw)
        if vf is None:
            continue
        path = ROOT / fc.FOLDER[day] / "videos" / tms[race]["file"]
        pts = video_pts(path, ROOT / "finish3" / "pts" / day / f"{race}.json")
        n = int((tw - FIRST) / STEP)
        want = {}
        for k in range(n, -1, -1):
            i = bisect.bisect_right(pts, vf - k * STEP + 1e-8) - 1
            if i >= 0:
                want[i] = round(k * STEP, 1)
        first, last = min(want), max(want)
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        i, pics = first, []
        while i <= last:
            ok, bgr = cap.read()
            if not ok:
                break
            if i in want:
                detection, boxes, P = riders(ident, bgr)
                pics.append({"to_finish": want[i], "t": pts[i], "frame": i, "detection": detection, "boxes": boxes, "probs": P})
            i += 1
        cap.release()
        res["races"][race] = {"video_finish": vf, "anchor": how, "clock_agrees": agree, "winner_time": tw, "clock_S": tms[race]["S"], "distance": off[race]["distance"], "pictures": pics}
        print(race, len(pics), "pictures, riders per picture", round(float(np.mean([len(p["boxes"]) for p in pics])), 1), flush=True)
        out_path.write_text(json.dumps(res))
    print("done", day, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
