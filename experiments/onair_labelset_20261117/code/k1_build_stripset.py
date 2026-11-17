"""Build a large, label-free evaluation set: every frame (3 s apart) whose running-order strip Qwen can read.

The strip read from the ORIGINAL frame is the scoring reference; the saved frame has the graphics masked,
exactly as the models see it. Split by race: development = odd races, test = even races (fixed here).
Writes stripset/frames/*.png and stripset/manifest.json.
"""
import bisect
import json
import sys
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SRC = Path("/home/ohead/ai-workspace/projects/onair_overnight_20261117")
sys.path.insert(0, str(ROOT / "code"))
from j1_strip_reader import ENDPOINT, MODEL, read_strip  # noqa: E402

OUT = ROOT / "stripset"
STEP = 3
STRIP = (480, 955, 1920, 1045)
DEV = {"R01", "R03", "R05", "R07", "R09"}

live = json.loads((ROOT / "evidence" / "live_segments.json").read_text())["races"]
masks = json.loads((SRC / "selection.json").read_text())["race_masks"]
vids = {v["race"]: v for v in json.loads((SRC / "evidence" / "video_sources.json").read_text())["videos"]}
(OUT / "frames").mkdir(parents=True, exist_ok=True)
manifest, tried = [], 0
for race in sorted(vids):
    pts = json.loads((SRC / "evidence" / f"{race}_pts.json").read_text())
    a, b = live[race]["live_start"] + 22, live[race]["live_end"] - 8
    cap = cv2.VideoCapture(str(SRC / vids[race]["file"]))
    kept = 0
    for t in range(a, b + 1, STEP):
        i = bisect.bisect_right(pts, t + 1e-8) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if not ok:
            continue
        tried += 1
        x1, y1, x2, y2 = STRIP
        nums, raw = read_strip(bgr[y1:y2, x1:x2])
        if nums is None:
            continue
        for mx1, my1, mx2, my2 in masks[race]:
            bgr[my1:my2, mx1:mx2] = 48
        fid = f"{race}_t{t:03d}"
        cv2.imwrite(str(OUT / "frames" / f"{fid}.png"), bgr)
        manifest.append({"frame": fid, "race": race, "pts": pts[i], "decoded_index": i, "strip": nums,
                         "split": "dev" if race in DEV else "test"})
        kept += 1
    cap.release()
    print(race, "kept", kept, flush=True)
(OUT / "manifest.json").write_text(json.dumps({
    "what": "frames 3 s apart whose running-order strip was read by Qwen (strip_reader_check.json: 37/38 exact, no wrong reads)",
    "split_rule": "development = R01 R03 R05 R07 R09, test = R02 R04 R06 R08 R10", "frames_tried": tried, "frames": manifest}, indent=1))
print("tried", tried, "kept", len(manifest), "dev", sum(m["split"] == "dev" for m in manifest), "test", sum(m["split"] == "test" for m in manifest))
try:
    urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/generate", data=json.dumps({"model": MODEL, "keep_alive": 0}).encode(), headers={"Content-Type": "application/json"}), timeout=60).read()
except Exception:
    pass
