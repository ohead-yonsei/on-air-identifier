"""Finish stretch, step 2b: a slit-scan ("photo finish") picture of the slow-motion replay, for reading the order across the line.
usage: fs2b_photofinish.py DAY_DIR [RACE ...]
A 5-pixel column just before the finish line is taken from every replay frame and the columns are laid side by side,
latest on the left, so horses face right and the first across the line is the rightmost. One picture per race."""
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
(FO / "photofinish").mkdir(parents=True, exist_ok=True)
times = json.loads((FO / "replay_times.json").read_text())
X, W, Y1, Y2, ROW = 1394, 5, 230, 660, 1500
for src in json.loads((O / "video_sources.json").read_text()):
    race = src["race"]
    if race not in times or times[race]["caption_start"] is None or (only and race not in only):
        continue
    tc = times[race]["caption_start"]
    pts = json.loads((O / "evidence" / f"{race}_pts.json").read_text())
    cap = cv2.VideoCapture(str(O / src["file"]))
    i0 = bisect.bisect_right(pts, tc - 3.0) - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    cols = []
    for i in range(i0, len(pts)):
        if pts[i] > tc + 18.0:
            break
        ok, bgr = cap.read()
        if not ok:
            break
        if pts[i] > tc + 1.0:   # stop when the replay caption has gone
            roi = cv2.cvtColor(bgr[985:1015, 420:1580], cv2.COLOR_BGR2HSV)
            if float(((roi[..., 1] < 30) & (roi[..., 2] > 225)).mean()) < 0.5:
                break
        cols.append(bgr[Y1:Y2, X:X + W].copy())
    cap.release()
    strip = np.hstack(cols[::-1])
    rows = []
    for k in range(strip.shape[1], 0, -ROW):   # first row = earliest crossings (right end of the strip)
        part = strip[:, max(0, k - ROW):k]
        if part.shape[1] < ROW:
            part = np.hstack([np.zeros((part.shape[0], ROW - part.shape[1], 3), np.uint8), part])
        rows.append(part)
    img = np.vstack(rows)
    cv2.putText(img, f"{race}: first across the line = right end of the top row", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(FO / "photofinish" / f"{race}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(race, "frames", len(cols), "picture", img.shape[1], "x", img.shape[0], flush=True)
