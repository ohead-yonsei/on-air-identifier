"""Finish stretch 2, step 3: run the identifier on the last 51 s of every race of a day, every 3 s, ending at the winner's finish.
usage: fn3_run.py DAY CONFIG [RACE ...]      DAY = otherday..otherday11 or 20261115, CONFIG = F2 | L1

The running-order strip is never read here. Time points: video time of the finish minus 51, 48, ..., 3, 0 s (the first two only feed the
carry rule; scoring uses the last 36 s). F2 = identify_l1.py --fast --fallback --leader-tau 0.5, L1 = its defaults.
Writes finish2/runs/CONFIG/DAY.json."""
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

day, config, only = sys.argv[1], sys.argv[2], set(sys.argv[3:])
assert config in ("F2", "L1")
ident = LeadingGroupIdentifier(fast=True, fallback=True, tau_leader=0.5) if config == "F2" else LeadingGroupIdentifier()
off, tms = fc.official(day), fc.times(day)
out_path = ROOT / "finish2" / "runs" / config / f"{day}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
(ROOT / "finish2" / "pts" / day).mkdir(parents=True, exist_ok=True)
res = {"day": day, "config": config, "races": {}}
if out_path.exists():          # keep the races already run (a rerun only adds the missing ones)
    old = json.loads(out_path.read_text())
    res["races"] = {r: v for r, v in old["races"].items() if "points" in v}
t0 = time.time()
for race in sorted(off):
    if only and race not in only:
        continue
    if "error" in off[race] or race in res["races"]:
        continue
    tw = fc.winner_time(off[race])
    vf, how, agree = fc.video_finish(tms.get(race), tw)
    if vf is None:
        res["races"][race] = {"skipped": how}
        print(race, "skipped:", how, flush=True)
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

    points = []
    for tau in range(51, -1, -3):
        t = vf - tau
        if t - 1.0 < pts[0]:
            continue
        bgr, i = grab(t)
        if bgr is None:
            continue
        pics = [bgr] + [grab(pts[i] - o)[0] for o in (0.5, 1.0)]
        r = ident.point(pts[i], [p for p in pics if p is not None], key=race)
        r["to_finish"] = tau
        points.append(r)
    cap.release()
    res["races"][race] = {"video_finish": vf, "anchor": how, "clock_agrees": agree, "winner_time": tw, "points": points}
    print(race, "finish at", vf, how, "|", " ".join(f"{p['to_finish']}:{p['statements']}" for p in points if p["to_finish"] <= 36), flush=True)
    out_path.write_text(json.dumps(res))
res["seconds"] = round(time.time() - t0, 1)
res["qwen_failures"] = ident.qwen_failures
out_path.write_text(json.dumps(res))
print("done", day, config, res["seconds"], "s", flush=True)
