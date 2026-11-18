"""Finish stretch 2, stage 2 data: riders in every picture of the last 37 s of each race, 5 pictures a second.
usage: fn6_dense.py DAY [RACE ...]

Same detection and cap classifier as the fast configuration (D0, then D3 when fewer than four riders; no Qwen), but the full
cap-colour probabilities of every rider are kept, so that rules that use many pictures can be tried without running the models again.
Writes finish2/dense/DAY.json: races -> pictures [{to_finish, t, detection, boxes, probs}]. The strip is never read."""
import bisect
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "identifier"))
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import identify2
from identify_l1 import MIN_RIDERS, LeadingGroupIdentifier, video_pts
from s3_classify import region as cap_region

SPAN, STEP = 37.0, 0.2


def riders(ident, bgr):
    b = bgr.copy()
    H, W = b.shape[:2]
    for x1, y1, x2, y2 in ident.masks:
        b[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = 48
    found, detection = identify2.find_riders(ident.det, b, ident.masks, "D0"), "D0"
    if len(found) < MIN_RIDERS:
        found, detection = identify2.find_riders(ident.det, b, ident.masks, "D3"), "D3"
    boxes = [[round(float(v), 1) for v in box] for box, _ in found]
    if not boxes:
        return detection, [], []
    rgb = Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
    xs = []
    for box in boxes:
        can = Image.new("RGB", (224, 224), (96, 96, 96))
        patch = ImageOps.contain(rgb.crop(cap_region(box, "R0_35", W, H)), (224, 224), Image.Resampling.LANCZOS)
        can.paste(patch, ((224 - patch.width) // 2, (224 - patch.height) // 2))
        xs.append(ident.tf(can))
    with torch.inference_mode():
        P = torch.softmax(ident.cnn(torch.stack(xs).cuda()), 1).cpu().numpy().round(4).tolist()
    return detection, boxes, P


def main():
    day, only = sys.argv[1], set(sys.argv[2:])
    ident = LeadingGroupIdentifier(fast=True)
    off, tms = fc.official(day), fc.times(day)
    out_path = ROOT / "finish2" / "dense" / f"{day}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "finish2" / "pts" / day).mkdir(parents=True, exist_ok=True)
    res = {"day": day, "span": SPAN, "step": STEP, "races": {}}
    if out_path.exists():
        res["races"] = json.loads(out_path.read_text())["races"]
    t0 = time.time()
    for race in sorted(off):
        if (only and race not in only) or "error" in off[race] or race in res["races"]:
            continue
        tw = fc.winner_time(off[race])
        vf, how, agree = fc.video_finish(tms.get(race), tw)
        if vf is None:
            continue
        path = ROOT / fc.FOLDER[day] / "videos" / tms[race]["file"]
        pts = video_pts(path, ROOT / "finish2" / "pts" / day / f"{race}.json")
        n = int(round(SPAN / STEP))
        want = [bisect.bisect_right(pts, vf - SPAN + k * STEP + 1e-8) - 1 for k in range(n + 1)]
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(want[0], 0))
        i, pics, wanted = max(want[0], 0), [], set(want)
        while i <= want[-1]:
            ok, bgr = cap.read()
            if not ok:
                break
            if i in wanted:
                detection, boxes, P = riders(ident, bgr)
                pics.append({"to_finish": round(vf - pts[i], 3), "t": pts[i], "frame": i, "detection": detection, "boxes": boxes, "probs": P})
            i += 1
        cap.release()
        res["races"][race] = {"video_finish": vf, "anchor": how, "clock_agrees": agree, "winner_time": tw, "pictures": pics}
        print(race, len(pics), "pictures, riders per picture", round(float(np.mean([len(p["boxes"]) for p in pics])), 1), flush=True)
        out_path.write_text(json.dumps(res))
    print("done", day, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
