"""far, exploration: contact sheets of finish pictures where the front rider is read confidently as a cap that is not the winner's.
usage: w10_finish_sheet.py OUT_DIR N [SEED]        (CPU; pictures stay on the server / scratch space, never in the repository)
Each tile: the picture 0.2 s before the finish, every rider box with the number read and its probability; the title gives the
official first three. Only races whose clock agrees and whose winner was 0.2 s or more clear."""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import w4_winner_guard as w4

DAYS = "otherday9 otherday10 otherday11 otherday14 otherday15 otherday16 otherday17 otherday18 otherday19 otherday20 otherday21 otherday22".split()
out, n = Path(sys.argv[1]), int(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
out.mkdir(parents=True, exist_ok=True)
cases = []
for d in DAYS:
    fc.FOLDER.setdefault(d, d)
    off, tms = fc.official(d), fc.times(d)
    for race, dr in sorted(w4.dense(d).items()):
        hs = sorted((h for h in off[race]["horses"] if h.get("rank") and h.get("time", {}).get("finish")), key=lambda h: h["rank"])
        if not dr["clock_agrees"] or len(hs) < 3 or hs[1]["time"]["finish"] - hs[0]["time"]["finish"] < 0.2:
            continue
        p = min((p for p in dr["pictures"] if p["boxes"]), key=lambda p: abs(p["to_finish"] - 0.2), default=None)
        if p is None or abs(p["to_finish"] - 0.2) > 0.15:
            continue
        b, P = np.array(p["boxes"], float), np.array(p["probs"], float)
        i = int(np.argmax((b[:, 0] + b[:, 2]) / 2))
        c = int(np.argmax(P[i]))
        if P[i, c] >= 0.9 and c != min(hs[0]["number"], 11) - 1:
            cases.append((d, race, p, [h["number"] for h in hs[:3]], ROOT / d / "videos" / tms[race]["file"]))
random.Random(seed).shuffle(cases)
print(len(cases), "cases; drawing", min(n, len(cases)))
tiles = []
for d, race, p, top3, path in cases[:n]:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, p["frame"])
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        continue
    P = np.array(p["probs"], float)
    for box, pr in zip(p["boxes"], P):
        x1, y1, x2, y2 = (int(v) for v in box)
        c = int(np.argmax(pr))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(bgr, f"{c + 1}:{pr[c]:.2f}", (x1, max(y1 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(bgr, f"{d} {race}  official 1-2-3: {top3}  ({p['to_finish']:.1f} s before the finish)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 5)
    cv2.putText(bgr, f"{d} {race}  official 1-2-3: {top3}  ({p['to_finish']:.1f} s before the finish)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)
    tiles.append(cv2.resize(bgr, (960, 540)))
for k in range(0, len(tiles), 4):
    group = tiles[k:k + 4]
    while len(group) < 4:
        group.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(group[:2]), np.hstack(group[2:])])
    cv2.imwrite(str(out / f"finish_{k // 4 + 1}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
print("wrote", (len(tiles) + 3) // 4, "sheets")
