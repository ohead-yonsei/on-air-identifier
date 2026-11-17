"""Run the identifier (detection D0 + Qwen 9B) on every strip frame and predecessor of a day. usage: dx_identify.py DAY_DIR
Same as d4_identify.py with the folder as an argument. The cap classifier is applied afterwards with wk2_apply.py."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / sys.argv[1]
man = json.loads((O / "manifest.json").read_text())
jobs = [(O / "out", [str(O / "frames" / f"{m['frame']}.png") for m in man["frames"]]),
        (O / "out_past", [str(O / "frames_past" / f"{p['name']}.png") for p in man["past"]])]
n = 0
for out, files in jobs:
    for k in range(0, len(files), 60):
        subprocess.run([sys.executable, str(ROOT / "identifier" / "identify2.py"), *files[k:k + 60], "--out", str(out), "--variant", "D0",
                        "--model", str(ROOT / "identifier" / "model.json"), "--no-overlay"], check=True)
        n += len(files[k:k + 60])
print("identified", n, "frames of", sys.argv[1])
