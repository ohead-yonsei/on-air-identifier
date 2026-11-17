"""Crop the broadcaster's running-order strip (bottom of the screen) for every labelled frame.

The strip is masked out of everything the models see. Here it is cut from the ORIGINAL video
frame at the same decoded index, only to be used as a scoring reference for "who is in front".
Writes leaders/strips/*.jpg, leaders/strip_sheet_*.jpg and leaders/strip_manifest.json.
"""
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = Path("/home/ohead/ai-workspace/projects/onair_overnight_20261117")
OUT = ROOT / "leaders"
STRIP = (480, 955, 1920, 1045)  # x1, y1, x2, y2 at 1920x1080
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def labelled_frames():
    out = {}
    for batch, ev, man in (("1", ROOT / "results" / "evidence.json", ROOT / "frame_manifest.json"),
                           ("2A", ROOT / "batch2" / "results" / "evidence.json", ROOT / "batch2" / "frame_manifest.json"),
                           ("2B", ROOT / "batch2b" / "results" / "evidence.json", ROOT / "batch2" / "frame_manifest.json")):
        frames = {r["frame"] for r in json.loads(ev.read_text()) if r["human_number"] and not r["duplicate_of"]}
        for m in json.loads(man.read_text()):
            if m["id"] in frames:
                out[m["id"]] = {"batch": batch, "race": m["race"], "decoded_index": m["decoded_index"], "pts": m["actual_pts"]}
    return out


vids = {v["race"]: v for v in json.loads((SRC / "evidence" / "video_sources.json").read_text())["videos"]}
frames = labelled_frames()
(OUT / "strips").mkdir(parents=True, exist_ok=True)
tiles, manifest = [], []
font = ImageFont.truetype(FONT, 20)
for fid in sorted(frames):
    f = frames[fid]
    cap = cv2.VideoCapture(str(SRC / vids[f["race"]]["file"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f["decoded_index"])
    ok, bgr = cap.read()
    cap.release()
    assert ok, fid
    x1, y1, x2, y2 = STRIP
    crop = bgr[y1:y2, x1:x2]
    cv2.imwrite(str(OUT / "strips" / f"{fid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((1080, 68), Image.LANCZOS)
    tile = Image.new("RGB", (1240, 72), "#101820")
    tile.paste(im, (160, 2))
    ImageDraw.Draw(tile).text((6, 24), fid, font=font, fill="white")
    tiles.append(tile)
    manifest.append({"frame": fid, **f, "strip_region_1920": STRIP})
(OUT / "strip_manifest.json").write_text(json.dumps(manifest, indent=1))
PER = 24
for k in range(0, len(tiles), PER):
    part = tiles[k:k + PER]
    sheet = Image.new("RGB", (1240, 72 * len(part)), (16, 24, 32))
    for i, t in enumerate(part):
        sheet.paste(t, (0, 72 * i))
    sheet.save(OUT / f"strip_sheet_{k // PER + 1}.jpg", quality=90)
print("frames", len(tiles), "sheets", (len(tiles) + PER - 1) // PER)
