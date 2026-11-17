"""For each finish-stretch frame of a day: is the running-order strip on screen (readable by the strip reader)?
usage: fs1b_strip_flag.py DAY_DIR  -> finish/DAY_DIR/strip_flags.json. See confirm/DESIGN_KO.md (the strip being absent is the proposed
run-time signal for "this is the finish stretch")."""
import bisect
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from j1_strip_reader import read_strip  # noqa: E402

day = sys.argv[1]
O, FO = ROOT / day, ROOT / "finish" / day
STRIP = (480, 955, 1920, 1045)
man = json.loads((FO / "manifest.json").read_text())
files = {s["race"]: s["file"] for s in json.loads((O / "video_sources.json").read_text())}
out, caps = {}, {}
for m in man["frames"]:
    race = m["race"]
    pts = json.loads((O / "evidence" / f"{race}_pts.json").read_text())
    if race not in caps:
        caps[race] = cv2.VideoCapture(str(O / files[race]))
    cap = caps[race]
    cap.set(cv2.CAP_PROP_POS_FRAMES, bisect.bisect_right(pts, m["pts"] + 1e-8) - 1)
    ok, bgr = cap.read()
    x1, y1, x2, y2 = STRIP
    nums, _ = read_strip(bgr[y1:y2, x1:x2])
    out[m["frame"]] = nums
(FO / "strip_flags.json").write_text(json.dumps(out, indent=1))
print(day, "finish frames", len(out), "with a readable strip", sum(v is not None for v in out.values()))
