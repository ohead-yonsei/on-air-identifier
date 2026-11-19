"""bench2: the identifier class on the whole race, every 3 s back from the finish, for the official track of the draft benchmark.
usage: bw1_whole.py DAY NAME [CAP_WEIGHTS]          writes bench2/whole/NAME/DAY.json

The same time points and the same call as fn10_whole.py (which gave finish2/whole/F2 = v9): fast flow, fallback, leader threshold 0.5.
CAP_WEIGHTS = cnn/models/caps2_a.pt gives v11 (the program's default since 2026-11-19). Only races whose clock agrees with the official
time are run (the official track has no other races). Frame times come from DAY/evidence/RACE_pts.json, as the benchmark's pictures do.
Nothing here reads the strip, the official result or the sound to decide what is said: the official time only places the time points."""
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
from identify_l1 import LeadingGroupIdentifier

day, name = sys.argv[1], sys.argv[2]
weights = ROOT / sys.argv[3] if len(sys.argv) > 3 else None
ident = LeadingGroupIdentifier(fast=True, fallback=True, tau_leader=0.5, cap_weights=weights)
off, tms = fc.official(day), fc.times(day)
out_path = ROOT / "bench2" / "whole" / name / f"{day}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
res = {"day": day, "config": name, "cap_weights": str(sys.argv[3]) if weights else "class default", "races": {}}
t0 = time.time()
for race in sorted(off):
    if "error" in off[race]:
        continue
    tw = fc.winner_time(off[race])
    vf, how, agree = fc.video_finish(tms.get(race), tw)
    if vf is None or not agree:
        continue
    pts = json.loads((ROOT / fc.FOLDER[day] / "evidence" / f"{race}_pts.json").read_text())
    cap = cv2.VideoCapture(str(ROOT / fc.FOLDER[day] / "videos" / tms[race]["file"]))

    def grab(t):
        i = bisect.bisect_right(pts, t + 1e-8) - 1
        if i < 0:
            return None, i
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        return (bgr if ok else None), i

    points = []
    for tau in range(3 * int((tw - 9.0) // 3), -1, -3):
        t = vf - tau
        if t - 1.0 < pts[0]:
            continue
        bgr, i = grab(t)
        if bgr is None:
            continue
        pics = [bgr] + [grab(pts[i] - o)[0] for o in (0.5, 1.0)]
        r = ident.point(pts[i], [p for p in pics if p is not None], key=race)
        r["to_finish"], r["race_time"] = tau, round(tw - tau, 1)
        points.append(r)
    cap.release()
    res["races"][race] = {"video_finish": vf, "anchor": how, "clock_agrees": agree, "winner_time": tw, "points": points}
    print(race, len(points), "points", flush=True)
    out_path.write_text(json.dumps(res))
res["seconds"] = round(time.time() - t0, 1)
out_path.write_text(json.dumps(res))
print("done", day, name, res["seconds"], "s", flush=True)
