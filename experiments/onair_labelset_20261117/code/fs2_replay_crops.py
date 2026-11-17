"""Finish stretch, step 2: crops of the slow-motion finish replay, for reading the order across the line.
usage: fs2_replay_crops.py DAY_DIR [RACE ...]
The replay carries a white caption box at the bottom; frames from 2 s before the caption appears to 7 s after, 1 s apart."""
import bisect
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
day, only = sys.argv[1], set(sys.argv[2:])
O = ROOT / day
FO = ROOT / "finish" / day
(FO / "replay_crops").mkdir(parents=True, exist_ok=True)
vids = {v["race"]: v for v in json.loads((FO / "manifest.json").read_text())["videos"]}
out = {}
for src in json.loads((O / "video_sources.json").read_text()):
    race = src["race"]
    if race not in vids or vids[race]["finish_time"] is None or (only and race not in only):
        continue
    F = vids[race]["finish_time"]
    pts = json.loads((O / "evidence" / f"{race}_pts.json").read_text())
    cap = cv2.VideoCapture(str(O / src["file"]))

    def grab(t):
        i = bisect.bisect_right(pts, t + 1e-8) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        return bgr if ok else None

    tc, t = None, F + 6.0
    while t < pts[-1] - 0.3:
        bgr = grab(t)
        if bgr is None:
            break
        roi = cv2.cvtColor(bgr[985:1015, 420:1580], cv2.COLOR_BGR2HSV)
        blu = cv2.cvtColor(bgr[985:1015, 295:355], cv2.COLOR_BGR2HSV)
        if float(((roi[..., 1] < 30) & (roi[..., 2] > 225)).mean()) > 0.45 and float(((blu[..., 0] >= 100) & (blu[..., 0] <= 125) & (blu[..., 1] > 120)).mean()) > 0.5:
            tc = t
            break
        t += 0.5
    out[race] = {"finish_time": F, "caption_start": tc}
    if tc is None:
        print(race, "no replay caption found", flush=True)
        continue
    tiles = []
    for k in range(-2, 8):
        bgr = grab(tc + k)
        if bgr is None:
            continue
        band = cv2.resize(bgr[110:610], (1600, 417))
        cv2.putText(band, f"{race} replay {k:+d}s", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(band)
    for part in range(0, len(tiles), 5):
        cv2.imwrite(str(FO / "replay_crops" / f"{race}_{part // 5 + 1}.jpg"), np.vstack(tiles[part:part + 5]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    # compact sheet: only the part of the picture near the line (right 60 %), two columns, for cheaper reading
    small = [cv2.resize(t[:, 560:], (780, 313)) for t in tiles]
    for k, t in enumerate(small):
        cv2.putText(t, f"{k - 2:+d}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    while len(small) % 2:
        small.append(np.zeros_like(small[0]))
    (FO / "replay_compact").mkdir(exist_ok=True)
    cv2.imwrite(str(FO / "replay_compact" / f"{race}.jpg"), np.vstack([np.hstack(small[k:k + 2]) for k in range(0, len(small), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cap.release()
    print(race, "finish", round(F, 1), "caption", tc, flush=True)
(FO / "replay_times.json").write_text(json.dumps(out, indent=1))
