"""Other-day diagnosis: draw the new identifier's labels on frames of chosen races, with the strip order in the caption.

usage: python od4_sheets.py R08 R10
Green = sure, orange = not sure. Caption: frame, strip top five, numbers the identifier would say (3 frames, front 2, sure x2).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "otherday"
man = json.loads((O / "manifest.json").read_text())
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
cap_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)


def label(name, past):
    riders = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
    P = json.loads((O / "cnn_probs" / f"{name}.json").read_text())["probs"]
    if not riders:
        return []
    P = np.array(P)
    Sc = P.copy()
    for i, r in enumerate(riders):
        if r["qwen_number"]:
            Sc[i, r["qwen_number"] - 1] += 1.0
    ri, cj = linear_sum_assignment(-Sc)
    a = {int(i): int(j) for i, j in zip(ri, cj)}
    out = []
    for i, r in enumerate(riders):
        j = a.get(i)
        no = 0 if j is None else j + 1
        out.append({"box": r["box"], "no": no, "sure": bool(j is not None and (r["qwen_number"] == no or P[i, j] >= 0.8)), "q": r["qwen_number"]})
    return out


for race in sys.argv[1:]:
    frames = [m for m in man["frames"] if m["race"] == race]
    pick = frames[:: max(1, len(frames) // 6)][:6]
    tiles = []
    for m in pick:
        votes = Counter()
        cur = None
        for name, past in ((m["frame"], False), (f"{m['frame']}_m05", True), (f"{m['frame']}_m10", True)):
            lab = label(name, past)
            if not past:
                cur = lab
            for n in {x["no"] for x in sorted(lab, key=lambda x: -x["box"][3])[:2] if x["sure"] and x["no"]}:
                votes[n] += 1
        said = sorted(n for n, v in votes.items() if v >= 2)
        im = Image.open(O / "frames" / f"{m['frame']}.png").convert("RGB")
        d = ImageDraw.Draw(im)
        for x in cur:
            col = "#00e676" if x["sure"] else "#ffa000"
            d.rectangle(x["box"], outline=col, width=4)
            d.text((x["box"][0], max(0, x["box"][1] - 30)), (str(x["no"]) if x["sure"] else f"{x['no']}?"), font=font, fill=col, stroke_fill="black", stroke_width=3)
        im = im.crop((0, 150, 1920, 930)).resize((960, 390), Image.LANCZOS)
        tile = Image.new("RGB", (960, 424), "#101820")
        tile.paste(im, (0, 34))
        ok = [n for n in said if n in m["strip"][:3]]
        ImageDraw.Draw(tile).text((8, 6), f"{m['frame']}  strip {m['strip']}  says {said}  correct {ok}", font=cap_font, fill="white")
        tiles.append(tile)
    sheet = Image.new("RGB", (1920, 424 * ((len(tiles) + 1) // 2)), (16, 24, 32))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * 960, (i // 2) * 424))
    sheet.save(O / f"sheet_{race}.jpg", quality=88)
    print(race, "frames", len(frames), "shown", len(tiles))
