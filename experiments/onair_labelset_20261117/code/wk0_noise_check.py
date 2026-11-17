"""How often is a front box really one of the strip's five? Checked with the hand labels of 2026-11-14 (20 frames).
Development check for weak/DESIGN_KO.md (strip-based partial labels)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "otherday"
strip = {m["frame"]: m["strip"] for m in json.loads((O / "manifest.json").read_text())["frames"]}
by = {}
for r in json.loads((O / "label" / "labels.json").read_text())["riders"]:
    by.setdefault(r["frame"], []).append(r)
for K in (1, 2, 3, 4, 5):
    n = ok = unk = 0
    for f, rs in by.items():
        for r in sorted(rs, key=lambda r: -r["person_box"][3])[:K]:
            if not r["human_number"]:
                unk += 1
                continue
            n += 1
            ok += r["human_number"] in strip[f]
    print(f"front {K} boxes: labelled {n}, in strip {ok} ({ok/n:.2f}), unlabelled {unk}")
