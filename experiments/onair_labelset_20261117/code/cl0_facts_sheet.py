"""AI commentary demo, check: a sheet with each race's info graphic and its clock at three moments, to verify the facts read by the model.
usage: cl0_facts_sheet.py DAY OUT.jpg"""
import bisect
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "live_commentary" / sys.argv[1]
rows = []
for res_path in sorted((O / "results").glob("R??.json")):
    race = res_path.stem
    res = json.loads(res_path.read_text())
    pts = json.loads(res_path.with_suffix(".pts.json").read_text())
    cap = cv2.VideoCapture(str(O / "videos" / f"{race}.mp4"))

    def grab(t):
        cap.set(cv2.CAP_PROP_POS_FRAMES, bisect.bisect_right(pts, t + 1e-8) - 1)
        return cap.read()[1]

    P = res["points"]
    tiles = [cv2.resize(grab(P[0]["t"])[0:210, 0:360], (360, 210))]
    for k in (0, len(P) // 2, len(P) - 1):
        c = grab(P[k]["t"])[930:1055, 70:360].copy()
        cv2.putText(c, "t=%d" % round(P[k]["t"]), (170, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(cv2.resize(c, (290, 210)))
    row = np.hstack(tiles)
    cv2.putText(row, race, (290, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    rows.append(row)
    cap.release()
cv2.imwrite(sys.argv[2], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(np.vstack(rows).shape)
