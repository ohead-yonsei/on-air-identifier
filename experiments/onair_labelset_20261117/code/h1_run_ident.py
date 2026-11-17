"""Run the identifier on every frame that has a running-order strip, with the model trained without that frame's race."""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
strip = json.loads((ROOT / "leaders" / "strip_order.json").read_text())["order"]
by_race = defaultdict(list)
for f in sorted(strip):
    p = ROOT / "frames" / f"{f}.png"
    if not p.exists():
        p = ROOT / "batch2" / "frames" / f"{f}.png"
    assert p.exists(), f
    by_race[f.split("_")[0]].append(str(p))
for race, frames in sorted(by_race.items()):
    subprocess.run([sys.executable, str(ROOT / "identifier" / "identify.py"), *frames, "--out", str(ROOT / "leaders" / "ident_out"),
                    "--masks", "sep06", "--model", str(ROOT / "identifier" / "models_loro" / f"excl_{race}.json")], check=True)
