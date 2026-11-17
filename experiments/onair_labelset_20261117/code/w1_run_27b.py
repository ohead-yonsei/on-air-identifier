"""Run the 27B identifier on the held-out stripset frames and their 0.5 s and 1.0 s predecessors."""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "stripset"
frames = [m for m in json.loads((S / "manifest.json").read_text())["frames"] if m["split"] == "test"]
jobs = defaultdict(lambda: defaultdict(list))  # out dir -> race -> files
for m in frames:
    jobs[S / "out27" / "D0"][m["race"]].append(str(S / "frames" / f"{m['frame']}.png"))
    for tag in ("m05", "m10"):
        jobs[S / "out_past27" / "D0"][m["race"]].append(str(S / "frames_past" / f"{m['frame']}_{tag}.png"))
for out, by_race in jobs.items():
    for race, fs in sorted(by_race.items()):
        subprocess.run([sys.executable, str(ROOT / "identifier" / "identify2_27b.py"), *fs, "--out", str(out), "--variant", "D0",
                        "--model", str(ROOT / "identifier" / "models_loro" / f"excl_{race}.json"), "--no-overlay"], check=True)
print("done", len(frames), "frames x 3")
