"""Extract the frames 0.5, 1.0 and 1.5 s BEFORE each stripset frame, and run the identifier on them.

See multi/DESIGN_KO.md. Past frames only. Writes stripset/frames_past/*.png and stripset/out_past/D0/*.json.
"""
import bisect
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SRC = Path("/home/ohead/ai-workspace/projects/onair_overnight_20261117")
S = ROOT / "stripset"
OFFSETS = {"m05": 0.5, "m10": 1.0, "m15": 1.5}
frames = json.loads((S / "manifest.json").read_text())["frames"]
masks = json.loads((SRC / "selection.json").read_text())["race_masks"]
vids = {v["race"]: v for v in json.loads((SRC / "evidence" / "video_sources.json").read_text())["videos"]}
(S / "frames_past").mkdir(exist_ok=True)
by_race, record = defaultdict(list), []
for race in sorted({m["race"] for m in frames}):
    pts = json.loads((SRC / "evidence" / f"{race}_pts.json").read_text())
    cap = cv2.VideoCapture(str(SRC / vids[race]["file"]))
    for m in [x for x in frames if x["race"] == race]:
        for tag, off in OFFSETS.items():
            name = f"{m['frame']}_{tag}"
            path = S / "frames_past" / f"{name}.png"
            i = bisect.bisect_right(pts, m["pts"] - off + 1e-8) - 1
            assert pts[i] <= m["pts"] - off + 1e-6 and pts[i] < m["pts"]
            if not path.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, bgr = cap.read()
                assert ok, name
                for x1, y1, x2, y2 in masks[race]:
                    bgr[y1:y2, x1:x2] = 48
                cv2.imwrite(str(path), bgr)
            by_race[race].append(str(path))
            record.append({"name": name, "frame": m["frame"], "offset_s": off, "pts": pts[i], "decoded_index": i})
    cap.release()
    print(race, "past frames", len(by_race[race]), flush=True)
(S / "past_manifest.json").write_text(json.dumps(record, indent=1))
for race, fs in sorted(by_race.items()):
    subprocess.run([sys.executable, str(ROOT / "identifier" / "identify2.py"), *fs, "--out", str(S / "out_past" / "D0"), "--variant", "D0",
                    "--model", str(ROOT / "identifier" / "models_loro" / f"excl_{race}.json"), "--no-overlay"], check=True)
print("done", len(record), "past frames;", len(list((S / "out_past" / "D0").glob("*.json"))), "identified")
