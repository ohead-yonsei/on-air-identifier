"""live1: check, without the GPU, the part of identifier/live_run.py that reads the race clock frame by frame.
usage: lv0_clock_check.py VIDEO [IDENT.json]     prints the start and finish found going forwards, how fast frames are read, and
(with IDENT.json of code/ls2_run.py) the start and stop found by the whole-video method."""
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "identifier"))
from live_run import Clock

cap, clock, n, seen = cv2.VideoCapture(sys.argv[1]), Clock(), 0, {}
t0 = time.perf_counter()
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    v = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    n += 1
    clock.feed(v, bgr)
    if clock.start is not None and "start" not in seen:
        seen["start"] = round(v, 3)
    if clock.finish is not None and "finish" not in seen:
        seen["finish"] = round(v, 3)
dt = time.perf_counter() - t0
print("frames", n, "video seconds", round(v, 1), "read in", round(dt, 1), "s =", round(n / dt), "frames a second")
print("start", clock.start, "recognised at", seen.get("start"), "| finish", clock.finish, "recognised at", seen.get("finish"))
if len(sys.argv) > 2:
    print("whole-video method:", json.loads(Path(sys.argv[2]).read_text()).get("clock"))
