"""Finish stretch, step 2c: pictures of the slow-motion replay at the moments a horse reaches the finish line.
usage: fs2c_arrivals.py DAY_DIR [RACE ...]
The replay repeats each picture several times, so only pictures that differ from the one before are used. A picture counts as an
arrival when the column just before the line stops looking like the empty track (start of a run), and every 8th picture while
the column stays occupied. The first eight are saved as crops around the line, 4 rows x 2 columns, in time order."""
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
(FO / "arrivals").mkdir(parents=True, exist_ok=True)
times = json.loads((FO / "replay_times.json").read_text())
X, Y1, Y2 = 1392, 150, 660
for src in json.loads((O / "video_sources.json").read_text()):
    race = src["race"]
    if race not in times or times[race]["caption_start"] is None or (only and race not in only):
        continue
    tc = times[race]["caption_start"]
    pts = json.loads((O / "evidence" / f"{race}_pts.json").read_text())
    cap = cv2.VideoCapture(str(O / src["file"]))
    i0 = bisect.bisect_right(pts, tc - 3.0) - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    uniq, prev = [], None
    for i in range(i0, len(pts)):
        if pts[i] > tc + 18.0:
            break
        ok, bgr = cap.read()
        if not ok:
            break
        roi = cv2.cvtColor(bgr[985:1015, 420:1580], cv2.COLOR_BGR2HSV)
        on = float(((roi[..., 1] < 30) & (roi[..., 2] > 225)).mean()) >= 0.4
        if pts[i] > tc + 1.0 and not on:
            break
        small = cv2.resize(bgr[230:660, 600:1700], (275, 108)).astype(np.int16)
        if prev is None or float(np.abs(small - prev).mean()) > 1.5:
            if cv2.cvtColor(bgr[300:640, 1398:1416], cv2.COLOR_BGR2GRAY).mean(axis=0).max() > 205:      # the drawn finish line is there (skips other cameras)
                uniq.append((pts[i], bgr[Y1:Y2, 700:1650].copy(), bgr[Y1:Y2, X:X + 6].astype(np.int16)))
            prev = small
    cap.release()
    if len(uniq) < 5:
        print(race, "too few replay pictures", len(uniq), flush=True)
        continue
    bg = np.median(np.stack([u[2] for u in uniq]), axis=0)
    occ = [int((np.abs(u[2] - bg).max(axis=2).mean(axis=1) > 35).sum()) >= 25 for u in uniq]
    events, run = [], 0
    for k, o in enumerate(occ):
        if o:
            if run % 8 == 0:
                events.append(k)
            run += 1
        else:
            run = 0
    events = events[:8]
    tiles = []
    for n, k in enumerate(events):
        t = uniq[k][1].copy()
        cv2.putText(t, f"{race} arrival {n + 1}  t={uniq[k][0] - tc:+.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    img = np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)])
    img = cv2.resize(img, (img.shape[1] * 3 // 4, img.shape[0] * 3 // 4))
    cv2.imwrite(str(FO / "arrivals" / f"{race}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(race, "replay pictures", len(uniq), "occupied", sum(occ), "arrivals shown", len(events), "size", img.shape[1], "x", img.shape[0], flush=True)
