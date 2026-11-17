"""Finish stretch, step 1: find the moment the race timer stops, save frames of the last seconds and replay sheets.
usage: fs1_finish_frames.py DAY_DIR      (otherday, otherday2, otherday3)

Finish time F = last moment the digits inside the timer graphic change. Frames saved at F-6.0, F-4.5, F-3.0, F-1.5 s
(and 0.5 s / 1.0 s before each), with the broadcast graphics masked as everywhere else. No strip is on screen there.
Replay sheets (frames after F, small) are for a person or Claude to read the order across the line."""
import bisect
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
day = sys.argv[1]
O = ROOT / day
FO = ROOT / "finish" / day
TIMER = (70, 930, 360, 1055)
MASKS = [(0, 0, 360, 210), (1710, 0, 1920, 180), (0, 930, 1920, 1080)]
OFFSETS = (6.0, 4.5, 3.0, 1.5)
for d in ("frames", "frames_past", "replay"):
    (FO / d).mkdir(parents=True, exist_ok=True)
frames, past, info = [], [], []
for src in json.loads((O / "video_sources.json").read_text()):
    race = src["race"]
    if src.get("live_end") is None:
        continue
    path = O / src["file"]
    pts = json.loads((O / "evidence" / f"{race}_pts.json").read_text())
    cap = cv2.VideoCapture(str(path))

    def grab(t):
        i = bisect.bisect_right(pts, t + 1e-8) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        return (bgr if ok else None), i

    def white(bgr, bb):
        x1, y1, x2, y2 = TIMER
        hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        m = (hsv[..., 1] < 60) & (hsv[..., 2] > 200)
        return m[bb[1]:bb[3], bb[0]:bb[2]]

    ref, _ = grab(src["live_end"] - 1)
    x1, y1, x2, y2 = TIMER
    hsv = cv2.cvtColor(ref[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] >= 100) & (hsv[..., 0] <= 125) & (hsv[..., 1] > 140) & (hsv[..., 2] > 90)
    ys, xs = np.where(blue)
    bb = (int(np.percentile(xs, 5)), int(np.percentile(ys, 5)), int(np.percentile(xs, 95)), int(np.percentile(ys, 95)))
    last_change, prev = None, None
    t = src["live_end"] - 20.0
    series = []
    while t <= src["live_end"] + 0.01:
        bgr, _ = grab(t)
        w = white(bgr, bb)
        if prev is not None:
            ch = float((w != prev).mean())
            series.append((round(t, 2), round(ch, 4)))
            if ch > 0.01:
                last_change = t
        prev = w
        t += 0.1
    F = last_change
    info.append({"race": race, "live_end": src["live_end"], "finish_time": F, "timer_box": bb})
    if F is None:
        print(race, "no finish time", flush=True)
        continue
    for off in OFFSETS:
        bgr, i = grab(F - off)
        fid = f"{race}_f{int(off * 10):02d}"
        for mx1, my1, mx2, my2 in MASKS:
            bgr[my1:my2, mx1:mx2] = 48
        cv2.imwrite(str(FO / "frames" / f"{fid}.png"), bgr)
        frames.append({"frame": fid, "race": race, "pts": pts[i], "seconds_before_finish": off})
        for tag, o2 in (("m05", 0.5), ("m10", 1.0)):
            pb, pi = grab(pts[i] - o2)
            for mx1, my1, mx2, my2 in MASKS:
                pb[my1:my2, mx1:mx2] = 48
            cv2.imwrite(str(FO / "frames_past" / f"{fid}_{tag}.png"), pb)
            past.append({"name": f"{fid}_{tag}", "frame": fid, "pts": pts[pi]})
    tiles = []
    t = F - 1.0
    while t < pts[-1] - 0.2 and len(tiles) < 40:
        bgr, _ = grab(t)
        if bgr is None:
            break
        small = cv2.resize(bgr, (640, 360))
        cv2.putText(small, f"{race} F{t - F:+.1f}s", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(small)
        t += 1.0
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
    cv2.imwrite(str(FO / "replay" / f"{race}_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    cap.release()
    print(race, "live_end", src["live_end"], "finish", round(F, 1), "video length", round(pts[-1], 1), flush=True)
(FO / "manifest.json").write_text(json.dumps({"day": day, "frames": frames, "past": past, "videos": info}, indent=1))
print("frames", len(frames), "past", len(past))
