"""finish3, program check of --leader-sure: the identifier class with the leader bar 0.4 and the finish-stretch leader (v10) on the whole race,
time points as g2_whole_v10.py, written in the same format so that g3_score.py scores it.
usage: g13_whole_leader_sure.py DAY [BAR] [CAP_WEIGHTS NAME]          writes finish3/whole/V10L/DAY.json (or finish3/whole/NAME/DAY.json)
The two days used (2026-08-22, 08-16) were already scored for this change with the picture data (g11_score2.py): this only checks that the
program's class gives the same numbers as that calculation. No winner call is made here."""
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
from identify_l1 import LATE_SECONDS, LeadingGroupIdentifier, video_pts

day = sys.argv[1]
bar = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
weights, name = (ROOT / sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else (None, "V10L")
fc.FOLDER.setdefault(day, day)
ident = LeadingGroupIdentifier(fast=True, fallback=True, tau_leader=0.5, finish_leader=True, leader_sure=bar, cap_weights=weights)
off, tms = fc.official(day), fc.times(day)
out_path = ROOT / "finish3" / "whole" / name / f"{day}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
(ROOT / "finish3" / "pts" / day).mkdir(parents=True, exist_ok=True)
res = {"day": day, "config": f"V10 with the leader bar {bar}", "races": {}}
t0 = time.time()
for race in sorted(off):
    if "error" in off[race]:
        continue
    tw = fc.winner_time(off[race])
    vf, how, agree = fc.video_finish(tms.get(race), tw)
    if vf is None or not agree:          # only the races of the main analysis are needed for the comparison
        continue
    path = ROOT / fc.FOLDER[day] / "videos" / tms[race]["file"]
    pts = video_pts(path, ROOT / "finish3" / "pts" / day / f"{race}.json")
    cap = cv2.VideoCapture(str(path))

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
        late = tau <= LATE_SECONDS
        r = ident.point(pts[i], [p for p in pics if p is not None], key=race, late=late)
        r.setdefault("leader_strip_model", r.get("leader"))
        r.update(to_finish=tau, race_time=round(tw - tau, 1), late=late)
        del r["riders_now"]
        points.append(r)
    cap.release()
    res["races"][race] = {"video_finish": vf, "anchor": how, "clock_agrees": agree, "winner_time": tw, "points": points, "winner_call": {}}
    print(race, len(points), "points", flush=True)
    out_path.write_text(json.dumps(res))
res["seconds"] = round(time.time() - t0, 1)
out_path.write_text(json.dumps(res))
print("done", day, res["seconds"], "s", flush=True)
