"""Stage 2: pick new live-race frames, detect every rider, and cut crops for labelling.

- Frames: 4 per race, evenly spaced inside the live segment found in stage 1, never within
  3 s of a frame the Codex run already used (so the two label sets do not duplicate).
- Riders: COCO Faster R-CNN person boxes that sit on a horse box and are tall enough to read.
- Output: frames/, crops/ (one upscaled upper-body crop per rider), sheets/ (per-frame contact
  sheet + box overlay for drafting), riders.json, frame_manifest.json.
Reads videos/masks from the Codex folder (read-only). Writes only inside this folder.
"""
import bisect
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path("/home/ohead/ai-workspace/projects/onair_overnight_20261117")
os.environ["TORCH_HOME"] = str(ROOT / "model_cache")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.ops import nms

FRAMES_PER_RACE = 4
MIN_RIDER_HEIGHT = 70          # px at 1920x1080
SKIP_AFTER_START = 22          # s after the timer appears (race starts ~7 s later; skips gate and split-screen shots)
SKIP_BEFORE_END = 8            # s: the timer stays ~2 s after the finish; the run-in is a distant wide shot
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wait_gpu():
    """Never touch other jobs: wait while the research lock is held or another compute process runs."""
    lock = Path("/home/ohead/icip2027_research_gpu.lock")
    while True:
        held = []
        if lock.exists():
            inode = str(lock.stat().st_ino)
            held = [s for s in Path("/proc/locks").read_text().splitlines() if any(t.endswith(":" + inode) for t in s.split())]
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip()
        apps = "\n".join(l for l in apps.splitlines() if "ollama" not in l.lower())
        if not held and not apps:
            return
        print("gpu busy, waiting", held, apps, flush=True)
        time.sleep(30)


assert shutil.disk_usage(ROOT).free > 20_000_000_000
live = json.loads((ROOT / "evidence" / "live_segments.json").read_text())["races"]
sel = json.loads((SRC / "selection.json").read_text())
masks = sel["race_masks"]
used = {}
for f in sel["frames"]:
    used.setdefault(f["race"], []).append(f["target_pts"])
videos = {v["race"]: v for v in json.loads((SRC / "evidence" / "video_sources.json").read_text())["videos"]}

# ---- frame selection (fixed rule, no looking at detections) ----
plan = []
for race in sorted(videos):
    seg = live[race]
    a, b = seg["live_start"] + SKIP_AFTER_START, seg["live_end"] - SKIP_BEFORE_END
    step = (b - a) / (FRAMES_PER_RACE - 1)
    for k in range(FRAMES_PER_RACE):
        t = round(a + k * step)
        while any(abs(t - u) < 3 for u in used[race]):
            t += 3
        plan.append({"race": race, "target_pts": t})

for d in ("frames", "crops", "sheets"):
    (ROOT / d).mkdir(exist_ok=True)
cache_src = SRC / "model_cache"
if cache_src.exists() and not (ROOT / "model_cache").exists():
    shutil.copytree(cache_src, ROOT / "model_cache")  # copy, so the Codex folder is never written to

wait_gpu()
weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval().cuda()
font, small = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 16)

manifest, riders = [], []
for p in plan:
    race, t = p["race"], p["target_pts"]
    fid = f"{race}_t{t:03d}"
    pts = json.loads((SRC / "evidence" / f"{race}_pts.json").read_text())
    i = bisect.bisect_right(pts, t + 1e-8) - 1
    cap = cv2.VideoCapture(str(SRC / videos[race]["file"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, bgr = cap.read()
    cap.release()
    assert ok
    for x1, y1, x2, y2 in masks[race]:
        bgr[y1:y2, x1:x2] = 48
    fpath = ROOT / "frames" / f"{fid}.png"
    cv2.imwrite(str(fpath), bgr)
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().cuda() / 255
    with torch.inference_mode():
        pred = model([tensor])[0]

    def boxes_of(cls):
        m = (pred["labels"] == cls) & (pred["scores"] >= 0.5)
        bx, sc = pred["boxes"][m], pred["scores"][m]
        keep = nms(bx, sc, 0.45)
        return list(zip(bx[keep].cpu().tolist(), sc[keep].cpu().tolist()))

    persons, horses = boxes_of(1), boxes_of(19)

    def inter(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    def area(a):
        return max(0, a[2] - a[0]) * max(0, a[3] - a[1])

    kept = []
    for box, score in persons:
        l, tp, r, btm = box
        if btm - tp < MIN_RIDER_HEIGHT:
            continue
        if l < 4 or tp < 4 or r > W - 4 or btm > H - 4:
            continue
        cx, cy = (l + r) / 2, (tp + btm) / 2
        if any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in masks[race]):
            continue
        if not any(inter(box, hb) / area(box) >= 0.15 for hb, _ in horses):
            continue
        kept.append((box, score))
    # drop a box that is mostly inside a larger kept box (arm / horse-head fragments)
    final = []
    for box, score in kept:
        if any(o is not box and area(o) > area(box) and inter(box, o) / area(box) > 0.7 for o, _ in kept):
            continue
        final.append((box, score))
    final.sort(key=lambda x: (x[0][0] + x[0][2]) / 2)

    im = Image.fromarray(rgb)
    overlay = im.copy()
    d = ImageDraw.Draw(overlay)
    tiles = []
    for j, (box, score) in enumerate(final):
        rid = f"{fid}_p{j:02d}"
        l, tp, r, btm = box
        h = btm - tp
        crop_box = [max(0, math.floor(l - 10)), max(0, math.floor(tp - 10)), min(W, math.ceil(r + 10)), min(H, math.ceil(tp + 0.55 * h + 6))]
        patch = im.crop(crop_box)
        scale = min(220 / patch.height, 260 / patch.width)
        patch = patch.resize((max(1, round(patch.width * scale)), max(1, round(patch.height * scale))), Image.LANCZOS)
        patch.save(ROOT / "crops" / f"{rid}.jpg", quality=92)
        riders.append({"id": rid, "race": race, "frame": fid, "pts": pts[i], "person_box": [round(x, 1) for x in box],
                       "height_px": round(h, 1), "score": round(score, 3), "crop_box": crop_box, "crop": f"crops/{rid}.jpg"})
        d.rectangle(box, outline="#00ff90", width=2)
        d.text((l, max(0, tp - 24)), f"p{j:02d}", font=font, fill="yellow", stroke_fill="black", stroke_width=2)
        tile = Image.new("RGB", (280, 260), "#1b2630")
        tile.paste(patch, ((280 - patch.width) // 2, 34))
        ImageDraw.Draw(tile).text((8, 6), f"p{j:02d}  h={h:.0f}px", font=small, fill="white")
        tiles.append(tile)
    overlay.resize((1280, 720)).save(ROOT / "sheets" / f"{fid}_boxes.jpg", quality=88)
    if tiles:
        cols = 6
        sheet = Image.new("RGB", (280 * cols, 260 * math.ceil(len(tiles) / cols)), (30, 30, 30))
        for k, tile in enumerate(tiles):
            sheet.paste(tile, ((k % cols) * 280, (k // cols) * 260))
        sheet.save(ROOT / "sheets" / f"{fid}_riders.jpg", quality=90)
    manifest.append({"id": fid, "race": race, "target_pts": t, "actual_pts": pts[i], "decoded_index": i,
                     "file": f"frames/{fid}.png", "sha256": sha(fpath), "masks": masks[race],
                     "persons_detected": len(persons), "horses_detected": len(horses), "riders_kept": len(final)})
    print(fid, "persons", len(persons), "horses", len(horses), "riders kept", len(final), flush=True)

(ROOT / "frame_manifest.json").write_text(json.dumps(manifest, indent=1))
(ROOT / "riders.json").write_text(json.dumps({
    "rule": {"frames_per_race": FRAMES_PER_RACE, "min_rider_height_px": MIN_RIDER_HEIGHT, "person_score": 0.5, "nms": 0.45,
             "on_horse": "person box overlaps a horse box by >=15% of its area", "weights": str(weights)},
    "riders": riders}, indent=1))
print("frames", len(manifest), "riders", len(riders))
