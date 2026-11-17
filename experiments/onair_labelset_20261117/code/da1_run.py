"""Adaptive detection (DA): keep the usual detection (D0); where it finds fewer than 4 riders in a picture, use the small-rider
detection (D3: score 0.3, height 40 px, zoom) for that picture instead. See da/DESIGN_KO.md.

usage: da1_run.py SRC_DIR NAME     SRC_DIR = otherday | otherday2 | otherday3 | finish/otherday3 ...   ->  da/NAME/{out,out_past}
Frames and manifest are linked from SRC_DIR. Then apply the cap classifier with: wk2_apply.py all_two_days.pt da/NAME cnn_probs_two"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src, name = sys.argv[1:3]
SRC = ROOT / src
O = ROOT / "da" / name
O.mkdir(parents=True, exist_ok=True)
MIN_RIDERS = 4
for f in ("frames", "frames_past", "manifest.json", "finish_order.json"):
    if (SRC / f).exists() and not (O / f).exists():
        os.symlink(SRC / f, O / f)
man = json.loads((O / "manifest.json").read_text())
stats = {}
for sub, frames_dir, names in (("out", "frames", [m["frame"] for m in man["frames"]]), ("out_past", "frames_past", [p["name"] for p in man["past"]])):
    (O / sub).mkdir(exist_ok=True)
    (O / f"{sub}_D3").mkdir(exist_ok=True)
    few = [n for n in names if len(json.loads((SRC / sub / f"{n}.json").read_text())["riders"]) < MIN_RIDERS]
    todo = [str(O / frames_dir / f"{n}.png") for n in few if not (O / f"{sub}_D3" / f"{n}.json").exists()]
    for k in range(0, len(todo), 60):
        subprocess.run([sys.executable, str(ROOT / "identifier" / "identify2.py"), *todo[k:k + 60], "--out", str(O / f"{sub}_D3"), "--variant", "D3",
                        "--model", str(ROOT / "identifier" / "model.json"), "--no-overlay"], check=True)
    for n in names:
        shutil.copyfile((O / f"{sub}_D3" / f"{n}.json") if n in set(few) else (SRC / sub / f"{n}.json"), O / sub / f"{n}.json")
    stats[sub] = {"pictures": len(names), "switched_to_D3": len(few)}
(O / "da_stats.json").write_text(json.dumps(stats, indent=1))
print(name, stats)
