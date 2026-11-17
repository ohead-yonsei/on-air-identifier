"""Step 1 of ON-AIR_GOAL.md: contact sheet of the frames where the strip's #1 horse has no human-labelled box."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
L = ROOT / "leaders"
frames = json.loads((L / "frames.json").read_text())
human = {}
for ev in (ROOT / "results" / "evidence.json", ROOT / "batch2" / "results" / "evidence.json", ROOT / "batch2b" / "results" / "evidence.json"):
    for r in json.loads(ev.read_text()):
        human.setdefault(r["frame"], []).append(r["human_number"] if not r["duplicate_of"] else None)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
miss = [f for f in frames if f["strip"][0] not in [n for n in human.get(f["frame"], []) if n]]
tiles = []
for f in miss:
    im = Image.open(L / "ident_out" / f"{f['frame']}_overlay.jpg").resize((960, 540), Image.LANCZOS)
    tile = Image.new("RGB", (960, 580), "#101820")
    tile.paste(im, (0, 40))
    nums = human.get(f["frame"], [])
    cap = f"{f['frame']}  strip {f['strip']}  labelled {sorted(n for n in nums if n)}  unclear boxes {sum(n is None for n in nums)}"
    ImageDraw.Draw(tile).text((8, 8), cap, font=font, fill="white")
    tiles.append(tile)
for k in range(0, len(tiles), 4):
    part = tiles[k:k + 4]
    sheet = Image.new("RGB", (1920, 580 * ((len(part) + 1) // 2)), (16, 24, 32))
    for i, t in enumerate(part):
        sheet.paste(t, ((i % 2) * 960, (i // 2) * 580))
    sheet.save(L / f"miss_sheet_{k // 4 + 1}.jpg", quality=90)
(L / "miss_frames.json").write_text(json.dumps([{"frame": f["frame"], "strip": f["strip"], "system_front3": f.get("system_front3")} for f in miss], indent=1))
print(len(miss), "frames:", [f["frame"] for f in miss])
