"""Other day: cut rider crops for labelling. Per race, the two strip frames with the most rider boxes (ties: earliest).

Writes otherday/label/crops/*.jpg, otherday/label/sheets/<frame>.jpg and otherday/label/riders.json.
The sheet caption shows the broadcaster's strip (top five numbers at that moment) as a labelling aid.
"""
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "otherday"
L = O / "label"
(L / "crops").mkdir(parents=True, exist_ok=True)
(L / "sheets").mkdir(exist_ok=True)
man = json.loads((O / "manifest.json").read_text())["frames"]
small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
cap = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
riders_out, picked = [], []
for race in sorted({m["race"] for m in man}):
    fr = [(len(json.loads((O / "out" / f"{m['frame']}.json").read_text())["riders"]), m) for m in man if m["race"] == race]
    fr.sort(key=lambda x: (-x[0], x[1]["pts"]))
    chosen = []
    for n, m in fr:  # keep the two richest frames at least 6 s apart so they are not near-duplicates
        if all(abs(m["pts"] - c["pts"]) >= 6 for c in chosen):
            chosen.append(m)
        if len(chosen) == 2:
            break
    picked += chosen
for m in sorted(picked, key=lambda m: m["frame"]):
    bgr = cv2.imread(str(O / "frames" / f"{m['frame']}.png"))
    H, W = bgr.shape[:2]
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    riders = sorted(json.loads((O / "out" / f"{m['frame']}.json").read_text())["riders"], key=lambda r: r["x"])
    tiles = []
    for j, r in enumerate(riders):
        l, t, rr, b = r["box"]
        h = b - t
        box = [max(0, math.floor(l - 10)), max(0, math.floor(t - 10)), min(W, math.ceil(rr + 10)), min(H, math.ceil(t + 0.55 * h + 6))]
        patch = im.crop(box)
        s = min(220 / patch.height, 260 / patch.width)
        patch = patch.resize((max(1, round(patch.width * s)), max(1, round(patch.height * s))), Image.LANCZOS)
        rid = f"{m['frame']}_p{j:02d}"
        patch.save(L / "crops" / f"{rid}.jpg", quality=92)
        riders_out.append({"id": rid, "race": m["race"], "frame": m["frame"], "pts": m["pts"], "person_box": r["box"], "height_px": r["h_px"]})
        tile = Image.new("RGB", (280, 260), "#1b2630")
        tile.paste(patch, ((280 - patch.width) // 2, 34))
        ImageDraw.Draw(tile).text((8, 6), f"p{j:02d}  h={h:.0f}px", font=small, fill="white")
        tiles.append(tile)
    cols = 6
    sheet = Image.new("RGB", (280 * cols, 30 + 260 * math.ceil(max(1, len(tiles)) / cols)), (30, 30, 30))
    ImageDraw.Draw(sheet).text((8, 4), f"{m['frame']}   strip top five: {m['strip']}", font=cap, fill="white")
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % cols) * 280, 30 + (k // cols) * 260))
    sheet.save(L / "sheets" / f"{m['frame']}.jpg", quality=90)
(L / "riders.json").write_text(json.dumps({"rule": "per race, the two strip frames with the most rider boxes, at least 6 s apart", "frames": [m["frame"] for m in picked], "riders": riders_out}, indent=1))
print("frames", len(picked), "riders", len(riders_out))
print(sorted((m["frame"], sum(r["frame"] == m["frame"] for r in riders_out)) for m in picked))
