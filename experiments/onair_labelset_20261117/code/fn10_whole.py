"""Finish stretch 2, stage 3: the fast configuration (F2) on the whole race, every 3 s, without the strip.
usage: fn10_whole.py DAY [RACE ...]

Time points: the winner's finish and every 3 s before it, back to 9 s after the start (the start = finish minus the official winning
time). Writes finish2/whole/F2/DAY.json. The part from 51 s before the finish on is also written in the format of fn3_run.py to
finish2/runs/F2/DAY.json when that file does not exist yet (identical time points; the earlier history does not change them)."""
import bisect
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "identifier"))
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
from identify_l1 import LeadingGroupIdentifier, video_pts

day, only = sys.argv[1], set(sys.argv[2:])
ident = LeadingGroupIdentifier(fast=True, fallback=True, tau_leader=0.5)
off, tms = fc.official(day), fc.times(day)
out_path = ROOT / "finish2" / "whole" / "F2" / f"{day}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
(ROOT / "finish2" / "pts" / day).mkdir(parents=True, exist_ok=True)
res = {"day": day, "config": "F2", "races": {}}
if out_path.exists():
    res["races"] = {r: v for r, v in json.loads(out_path.read_text())["races"].items() if "points" in v}
t0 = time.time()
for race in sorted(off):
    if (only and race not in only) or "error" in off[race] or race in res["races"]:
        continue
    tw = fc.winner_time(off[race])
    vf, how, agree = fc.video_finish(tms.get(race), tw)
    if vf is None:
        continue
    path = ROOT / fc.FOLDER[day] / "videos" / tms[race]["file"]
    pts = video_pts(path, ROOT / "finish2" / "pts" / day / f"{race}.json")
    cap = cv2.VideoCapture(str(path))

    def grab(t):
        i = bisect.bisect_right(pts, t + 1e-8) - 1
        if i < 0:
            return None, i
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        return (bgr if ok else None), i

    n = int((tw - 9.0) // 3)
    points = []
    for tau in range(3 * n, -1, -3):
        t = vf - tau
        if t - 1.0 < pts[0]:
            continue
        bgr, i = grab(t)
        if bgr is None:
            continue
        pics = [bgr] + [grab(pts[i] - o)[0] for o in (0.5, 1.0)]
        r = ident.point(pts[i], [p for p in pics if p is not None], key=race)
        r["to_finish"] = tau
        r["race_time"] = round(tw - tau, 1)
        points.append(r)
    cap.release()
    res["races"][race] = {"video_finish": vf, "anchor": how, "clock_agrees": agree, "winner_time": tw, "points": points}
    print(race, len(points), "points, first at race time", points[0]["race_time"] if points else None, flush=True)
    out_path.write_text(json.dumps(res))
res["seconds"] = round(time.time() - t0, 1)
out_path.write_text(json.dumps(res))
short = ROOT / "finish2" / "runs" / "F2" / f"{day}.json"
if not short.exists():
    short.parent.mkdir(parents=True, exist_ok=True)
    short.write_text(json.dumps({"day": day, "config": "F2", "from": "fn10_whole.py", "races": {r: {**v, "points": [p for p in v["points"] if p["to_finish"] <= 51]} for r, v in res["races"].items()}}))
print("done", day, res["seconds"], "s", flush=True)
