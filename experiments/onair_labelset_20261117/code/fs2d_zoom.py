"""Finish stretch reading aid: enlarged crops of the replay. usage: fs2d_zoom.py DAY RACE OUT.jpg T:X1,Y1,X2,Y2 [T:X1,Y1,X2,Y2 ...]
T = seconds after the replay caption appears; box in full-picture pixels. Crops are enlarged 3x and placed side by side."""
import bisect
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
day, race, out = sys.argv[1:4]
pts = json.loads((ROOT / day / "evidence" / f"{race}_pts.json").read_text())
tc = json.loads((ROOT / "finish" / day / "replay_times.json").read_text())[race]["caption_start"]
cap = cv2.VideoCapture(glob.glob(str(ROOT / day / "videos" / f"{race}_*.mp4"))[0])
tiles = []
for spec in sys.argv[4:]:
    t, box = spec.split(":")
    x1, y1, x2, y2 = [int(v) for v in box.split(",")]
    cap.set(cv2.CAP_PROP_POS_FRAMES, bisect.bisect_right(pts, tc + float(t)) - 1)
    ok, bgr = cap.read()
    tiles.append(cv2.resize(bgr[y1:y2, x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
h = max(t.shape[0] for t in tiles)
tiles = [np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
(ROOT / "finish" / day / "zoom").mkdir(exist_ok=True)
cv2.imwrite(str(ROOT / "finish" / day / "zoom" / out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 92])
