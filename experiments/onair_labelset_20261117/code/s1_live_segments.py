"""Stage 1: find the live-race segment of each 2026-09-06 video from the on-screen race timer.

The broadcast shows a timer box (blue outline, bottom-left) only while the race is running.
We sample one frame per second and measure the share of saturated-blue pixels in the timer
area. Reads the videos from the Codex experiment folder (read-only); writes only here.
"""
import bisect
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = Path("/home/ohead/ai-workspace/projects/onair_overnight_20261117")
ROI = (70, 930, 360, 1055)  # x1, y1, x2, y2 in 1920x1080
THRESH = 0.06

videos = json.loads((SRC / "evidence" / "video_sources.json").read_text())["videos"]
out = {}
for v in videos:
    race = v["race"]
    pts = json.loads((SRC / "evidence" / f"{race}_pts.json").read_text())
    cap = cv2.VideoCapture(str(SRC / v["file"]))
    series = []
    for t in range(1, int(pts[-1])):
        i = bisect.bisect_right(pts, t + 1e-8) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if not ok:
            break
        x1, y1, x2, y2 = ROI
        hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        h, s, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        blue = float(((h >= 100) & (h <= 125) & (s > 140) & (val > 90)).mean())
        white = float(((s < 40) & (val > 215)).mean())
        series.append({"t": t, "blue": round(blue, 4), "white": round(white, 4)})
    cap.release()
    on = [x["t"] for x in series if x["blue"] >= THRESH and x["white"] >= 0.15]
    # longest run of consecutive seconds (allow 2 s gaps for camera-cut transitions)
    runs, cur = [], []
    for t in on:
        if cur and t - cur[-1] > 3:
            runs.append(cur)
            cur = []
        cur.append(t)
    if cur:
        runs.append(cur)
    best = max(runs, key=len) if runs else []
    out[race] = {"duration": pts[-1], "timer_on_seconds": on,
                 "live_start": best[0] if best else None, "live_end": best[-1] if best else None,
                 "series": series}
    print(race, f"dur {pts[-1]:.1f}", "live", (best[0], best[-1]) if best else None, "runs", [(r[0], r[-1]) for r in runs])

(ROOT / "evidence").mkdir(parents=True, exist_ok=True)
(ROOT / "evidence" / "live_segments.json").write_text(json.dumps(
    {"roi_1920": ROI, "blue_threshold": THRESH, "method": "saturated-blue share in the race-timer area, 1 frame per second",
     "races": out}, indent=1))
