"""far, exploration: contact sheet of chosen moments with the rider boxes and the numbers read.
usage: w11_sheet_pick.py OUT.jpg DAY:RACE:TO_FINISH [...]      (CPU; pictures stay on the server / scratch space, never in the repository)"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import w4_winner_guard as w4

tiles = []
for spec in sys.argv[2:]:
    d, race, tf = spec.split(":")
    fc.FOLDER.setdefault(d, d)
    off, tms = fc.official(d), fc.times(d)
    dr = w4.dense(d)[race]
    hs = sorted((h for h in off[race]["horses"] if h.get("rank")), key=lambda h: h["rank"])
    p = min(dr["pictures"], key=lambda p: abs(p["to_finish"] - float(tf)))
    cap = cv2.VideoCapture(str(ROOT / d / "videos" / tms[race]["file"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, p["frame"])
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        continue
    for box, pr in zip(p["boxes"], np.array(p["probs"], float).reshape(-1, 11)):
        x1, y1, x2, y2 = (int(v) for v in box)
        c = int(np.argmax(pr))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(bgr, f"{c + 1}:{pr[c]:.2f}", (x1, max(y1 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    for col, th in (((255, 255, 255), 5), ((0, 0, 0), 2)):
        cv2.putText(bgr, f"{d} {race}  1-2-3: {[h['number'] for h in hs[:3]]}  {p['to_finish']:.1f} s to go", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, th)
    tiles.append(cv2.resize(bgr, (960, 540)))
rows = []
for k in range(0, len(tiles), 3):
    g = tiles[k:k + 3]
    while len(g) < 3:
        g.append(np.zeros_like(tiles[0]))
    rows.append(np.hstack(g))
cv2.imwrite(sys.argv[1], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("wrote", sys.argv[1], len(tiles), "tiles")
