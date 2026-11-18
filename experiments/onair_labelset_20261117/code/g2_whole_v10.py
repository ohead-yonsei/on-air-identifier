"""finish3: v10 on the whole race, every 3 s, without the strip. Time points as fn10_whole.py (the winner's finish and every 3 s
before it, back to 9 s after the start), so the numbers compare with the v9 whole-race measurement of finish2.
usage: g2_whole_v10.py DAY [--leader MODEL.joblib --name NAME] [RACE ...]

v10 = v9 (F2) + in the last 36 s the leader from the model learned on official times + winner call from the 31 pictures of the last 6 s
(identifier/identify_l1.py, class LeadingGroupIdentifier with finish_leader=True; the same code the program runs with --v10).
For every late time point the strip-trained model's leader is kept too ("leader_strip_model"), so v9 and v10 are compared on the same points.
The winner call is made twice: with the finish from the official time (as the experiments of finish2) and with the finish from the
clock stop alone (as the program does).
--leader/--name (stage 2 of finish3): another leader model is used at EVERY time point (late or not) and the run is written under NAME.
Writes finish3/whole/NAME/DAY.json (NAME = V10 by default). Nothing here reads the strip."""
import bisect
import json
import sys
import time
from pathlib import Path

import cv2
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "identifier"))
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
from identify_l1 import LATE_SECONDS, WINNER_SPAN, LeadingGroupIdentifier, video_pts

args = sys.argv[1:]
day = args.pop(0)
name, other = "V10", None
while args and args[0].startswith("--"):
    k = args.pop(0)
    if k == "--leader":
        other = args.pop(0)
    elif k == "--name":
        name = args.pop(0)
only = set(args)
fc.FOLDER.setdefault(day, day)
ident = LeadingGroupIdentifier(fast=True, fallback=True, tau_leader=0.5, finish_leader=True)
if other:
    ident.leader_fs = joblib.load(ROOT / other)
off, tms = fc.official(day), fc.times(day)
out_path = ROOT / "finish3" / "whole" / name / f"{day}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
(ROOT / "finish3" / "pts" / day).mkdir(parents=True, exist_ok=True)
res = {"day": day, "config": name, "leader_model": other or "finish2/models/fs_leader_all.joblib in the last 36 s", "races": {}}
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
    pts = video_pts(path, ROOT / "finish3" / "pts" / day / f"{race}.json")
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
        late = True if other else tau <= LATE_SECONDS
        r = ident.point(pts[i], [p for p in pics if p is not None], key=race, late=late)
        r.setdefault("leader_strip_model", r.get("leader"))          # not late: both are the strip-trained model's leader
        r.update(to_finish=tau, race_time=round(tw - tau, 1), late=late)
        del r["riders_now"]
        points.append(r)
    calls = {}
    for key, fin in (("official_anchor", vf), ("clock_stop", tms[race]["F"] - 0.2)):
        number, count = ident.winner_call([b for b in (grab(fin - WINNER_SPAN + 0.2 * k)[0] for k in range(int(WINNER_SPAN * 5) + 1)) if b is not None])
        calls[key] = {"number": number, "pictures_naming_each": count, "finish": round(fin, 3)}
    cap.release()
    res["races"][race] = {"video_finish": vf, "anchor": how, "clock_agrees": agree, "winner_time": tw, "points": points, "winner_call": calls}
    print(race, len(points), "points; winner call", calls["official_anchor"]["number"], calls["clock_stop"]["number"], flush=True)
    out_path.write_text(json.dumps(res))
res["seconds"] = round(time.time() - t0, 1)
out_path.write_text(json.dumps(res))
print("done", day, res["seconds"], "s", flush=True)
