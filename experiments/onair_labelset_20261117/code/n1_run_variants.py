"""Run identify2.py for the given split and variants, each frame with the model trained without its race.

usage: python n1_run_variants.py dev|test D0 D1 ...
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
split, variants = sys.argv[1], sys.argv[2:]
frames = [m for m in json.loads((ROOT / "stripset" / "manifest.json").read_text())["frames"] if m["split"] == split]
by_race = defaultdict(list)
for m in frames:
    by_race[m["race"]].append(str(ROOT / "stripset" / "frames" / f"{m['frame']}.png"))
for v in variants:
    for race, fs in sorted(by_race.items()):
        subprocess.run([sys.executable, str(ROOT / "identifier" / "identify2.py"), *fs, "--out", str(ROOT / "stripset" / "out" / v),
                        "--variant", v, "--model", str(ROOT / "identifier" / "models_loro" / f"excl_{race}.json"), "--no-overlay"], check=True)
print("done", split, variants, "frames", len(frames))
