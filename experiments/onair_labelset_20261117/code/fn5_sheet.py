"""Finish stretch 2: contact sheet of one race's last 36 s with the identifier's riders drawn (for looking; development days only).
usage: fn5_sheet.py DAY CONFIG RACE OUT.jpg [TILE_WIDTH]"""
import bisect
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn_common as fc

day, config, race, out = sys.argv[1:5]
tw = int(sys.argv[5]) if len(sys.argv) > 5 else 960
run = json.loads((fc.ROOT / "finish2" / "runs" / config / f"{day}.json").read_text())["races"][race]
off = fc.official(day)[race]
tm = fc.times(day)[race]
pts = json.loads((fc.ROOT / "finish2" / "pts" / day / f"{race}.json").read_text())
cap = cv2.VideoCapture(str(fc.ROOT / fc.FOLDER[day] / "videos" / tm["file"]))
tiles = []
for p in run["points"]:
    if p["to_finish"] > 36:
        continue
    i = bisect.bisect_right(pts, p["t"] + 1e-8) - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, b = cap.read()
    for r in p["riders_now"]:
        x1, y1, x2, y2 = map(int, r["box"])
        col = (0, 255, 0) if r["sure"] else (0, 165, 255)
        cv2.rectangle(b, (x1, y1), (x2, y2), col, 2)
        cv2.putText(b, str(r["number"]), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
    ref = fc.reference(off, run["winner_time"] - p["to_finish"])
    cv2.rectangle(b, (380, 0), (1920, 60), (0, 0, 0), -1)
    text = "-%ds said %s lead %s | ref %s" % (p["to_finish"], p["statements"], p.get("leader"), ref["order"][:4])
    cv2.putText(b, text, (390, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
    tiles.append(cv2.resize(b, (tw, tw * 9 // 16)))
while len(tiles) % 2:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
