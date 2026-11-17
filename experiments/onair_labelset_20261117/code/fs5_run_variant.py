"""Finish stretch, development: run the identifier with another detection variant on a day's finish frames.
usage: fs5_run_variant.py DAY_DIR VARIANT      -> finish_VARIANT/DAY_DIR/{out,out_past}  (frames are linked from finish/DAY_DIR)"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
day, variant = sys.argv[1:3]
SRC = ROOT / "finish" / day
O = ROOT / f"finish_{variant}" / day
O.mkdir(parents=True, exist_ok=True)
for name in ("frames", "frames_past", "manifest.json", "finish_order.json"):
    if not (O / name).exists():
        os.symlink(SRC / name, O / name)
man = json.loads((O / "manifest.json").read_text())
jobs = [(O / "out", [str(O / "frames" / f"{m['frame']}.png") for m in man["frames"]]),
        (O / "out_past", [str(O / "frames_past" / f"{p['name']}.png") for p in man["past"]])]
n = 0
for out, files in jobs:
    for k in range(0, len(files), 60):
        subprocess.run([sys.executable, str(ROOT / "identifier" / "identify2.py"), *files[k:k + 60], "--out", str(out), "--variant", variant,
                        "--model", str(ROOT / "identifier" / "model.json"), "--no-overlay"], check=True)
        n += len(files[k:k + 60])
print("identified", n, "finish-stretch frames of", day, "with", variant)
