"""ON-AIR identifier v5 (draft): which horse number is each visible jockey?

For one broadcast frame:
  1. find riders (COCO Faster R-CNN: a person box sitting on a horse box),
  2. read each cap twice: Qwen 9B names a colour; a small learned classifier scores all 11 numbers,
  3. give every rider a different number so that the total score is highest,
  4. mark each answer:
       sure    Qwen named a number and the assignment kept it   (about 88% right in our tests)
       guess   a number was assigned some other way             (about 37% right in our tests)
       unknown nothing usable
Output is one JSON per frame in the shape of the project's visible.json (no, state, score, x, h_px, box)
plus an overlay picture.

usage:
  python identify.py FRAME.png [FRAME2.png ...] --out OUTDIR [--masks sep06|oct10|none] [--field 1,2,3,...]
"""
import argparse
import base64
import io
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ["TORCH_HOME"] = str(ROOT / "model_cache")
sys.path.insert(0, str(ROOT / "code"))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.ops import nms

from s3_classify import ENDPOINT, MODEL, PROTOCOL, features, region  # same regions, features and Qwen protocol as the tests

W_QWEN = 1.0
MASKS = {  # broadcast graphics, as (x1, y1, x2, y2) fractions of the frame
    "sep06": [(0, 0, 360 / 1920, 210 / 1080), (1710 / 1920, 0, 1, 180 / 1080), (0, 930 / 1080, 1, 1)],
    "oct10": [(0, 0, 240 / 1280, 110 / 720), (0, 610 / 720, 1, 1)],
    "none": [],
}
COLOUR = {1: "white", 2: "yellow", 3: "red", 4: "black", 5: "blue", 6: "green", 7: "brown", 8: "pink", 9: "purple", 10: "sky blue", 11: "sky blue/white stripes"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def wait_gpu():
    lock = Path("/home/ohead/icip2027_research_gpu.lock")
    while True:
        held = []
        if lock.exists():
            inode = str(lock.stat().st_ino)
            held = [s for s in Path("/proc/locks").read_text().splitlines() if any(t.endswith(":" + inode) for t in s.split())]
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
        if not held and not [a for a in apps.strip().splitlines() if a and "ollama" not in a.lower()]:
            return
        print("gpu busy, waiting", flush=True)
        time.sleep(30)


class ColourModel:
    def __init__(self, path):
        m = json.loads(Path(path).read_text())
        self.classes = np.array(m["classes"])
        self.mean, self.scale = np.array(m["mean"]), np.array(m["scale"])
        self.coef, self.intercept = np.array(m["coef"]), np.array(m["intercept"])

    def proba(self, feat):
        p = np.zeros(11)
        if feat is None:
            return p
        z = ((np.array(feat) - self.mean) / self.scale) @ self.coef.T + self.intercept
        e = np.exp(z - z.max())
        p[self.classes - 1] = e / e.sum()
        return p


def find_riders(model, bgr, masks):
    """Same rule as the label-set extraction: person on a horse, tall enough, not on graphics or the frame edge."""
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

    inter = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda a: max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    persons, horses = boxes_of(1), boxes_of(19)
    min_h = 70 * W / 1920
    kept = []
    for box, score in persons:
        l, t, r, b = box
        if b - t < min_h or l < 4 or t < 4 or r > W - 4 or b > H - 4:
            continue
        cx, cy = (l + r) / 2, (t + b) / 2
        if any(x1 * W <= cx <= x2 * W and y1 * H <= cy <= y2 * H for x1, y1, x2, y2 in masks):
            continue
        if not any(inter(box, hb) / area(box) >= 0.15 for hb, _ in horses):
            continue
        kept.append((box, score))
    out = [(box, score) for box, score in kept
           if not any(o is not box and area(o) > area(box) and inter(box, o) / area(box) > 0.7 for o, _ in kept)]
    return sorted(out, key=lambda x: (x[0][0] + x[0][2]) / 2)


def ask_qwen(bgr, box, log):
    crop = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).crop(box)
    can = Image.new("RGB", (224, 224), (96, 96, 96))
    patch = ImageOps.contain(crop, (224, 224), Image.Resampling.LANCZOS)
    can.paste(patch, ((224 - patch.width) // 2, (224 - patch.height) // 2))
    buf = io.BytesIO()
    can.save(buf, format="PNG")
    msgs = [{"role": "system", "content": PROTOCOL["SYSTEM"]},
            {"role": "user", "content": PROTOCOL["PROMPT"], "images": [base64.b64encode(buf.getvalue()).decode()]}]
    body = json.dumps({"model": MODEL, "messages": msgs, "format": PROTOCOL["SCHEMA"], "stream": False, "think": False,
                       "keep_alive": "10m", "options": PROTOCOL["options"]}).encode()
    start = time.perf_counter()
    try:
        req = urllib.request.Request(ENDPOINT + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            raw = r.read().decode()
        cls = json.loads(json.loads(raw)["message"]["content"])["color_class"]
        number = PROTOCOL["CLASSES"].index(cls)
        err = None
    except Exception as e:  # a failed call counts as "Qwen did not answer"
        cls, number, err = None, 0, repr(e)
    log.append({"colour": cls, "number": number, "ms": round((time.perf_counter() - start) * 1000, 1), "error": err})
    return number


def identify(bgr, det_model, colour_model, masks, field, qlog):
    H, W = bgr.shape[:2]
    for x1, y1, x2, y2 in masks:  # hide graphics from the detector too
        bgr[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = 48
    riders = find_riders(det_model, bgr, masks)
    if not riders:
        return []
    P, Q = [], []
    for box, _ in riders:
        P.append(colour_model.proba(features(bgr, region(box, "R0_27", W, H), box)))
        Q.append(ask_qwen(bgr, region(box, "R0_35", W, H), qlog))
    P = np.array(P)
    S = P.copy()
    for i, q in enumerate(Q):
        if q:
            S[i, q - 1] += W_QWEN
    allowed = np.array([(n in field) for n in range(1, 12)])
    S[:, ~allowed] = -1.0  # numbers that are not in this race can never be assigned
    ri, cj = linear_sum_assignment(-S)
    assigned = {int(i): int(j) for i, j in zip(ri, cj)}
    out = []
    for i, (box, det_score) in enumerate(riders):
        j = assigned.get(i)
        if j is None or S[i, j] <= 0:
            no, state, score = 0, "unknown", 0.0
        else:
            no = j + 1
            state = "sure" if (Q[i] != 0 and Q[i] == no) else "guess"
            score = float(S[i, j])
        l, t, r, b = box
        out.append({"no": no, "state": state, "score": round(score, 4), "x": round((l + r) / 2, 1), "h_px": round(b - t, 1),
                    "box": [round(v, 1) for v in box], "colour": COLOUR.get(no), "qwen_number": int(Q[i]),
                    "p_learned": round(float(P[i, j]), 4) if j is not None else None,
                    "sure_strict": bool(state == "sure" and P[i, j] >= 0.1), "detector_score": round(det_score, 3)})
    return out


def draw(bgr, result, path):
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, max(16, im.width // 70))
    col = {"sure": "#00e676", "guess": "#ffa000", "unknown": "#9e9e9e"}
    for r in result:
        l, t, _, _ = r["box"]
        d.rectangle(r["box"], outline=col[r["state"]], width=3)
        label = "?" if r["no"] == 0 else (str(r["no"]) if r["state"] == "sure" else f'{r["no"]}?')
        d.text((l, max(0, t - font.size - 4)), label, font=font, fill=col[r["state"]], stroke_fill="black", stroke_width=3)
    im.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--masks", default="sep06", choices=list(MASKS))
    ap.add_argument("--field", default=",".join(str(n) for n in range(1, 12)), help="numbers running in this race, e.g. 1,2,3,4,5,6,7,8,9")
    ap.add_argument("--model", default=str(HERE / "model.json"), help="colour-classifier JSON (default: trained on all labelled riders)")
    a = ap.parse_args()
    field = {int(x) for x in a.field.split(",")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wait_gpu()
    det = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().cuda()
    cm = ColourModel(a.model)
    for f in a.frames:
        bgr = cv2.imread(f)
        qlog, start = [], time.perf_counter()
        res = identify(bgr, det, cm, MASKS[a.masks], field, qlog)
        total_ms = (time.perf_counter() - start) * 1000
        name = Path(f).stem
        (out / f"{name}.json").write_text(json.dumps({
            "frame": f, "width": bgr.shape[1], "height": bgr.shape[0], "masks": a.masks, "field": sorted(field), "model": a.model,
            "riders": res, "qwen_calls": qlog, "total_ms": round(total_ms, 1)}, indent=1))
        draw(bgr, res, out / f"{name}_overlay.jpg")
        k = {s: sum(r["state"] == s for r in res) for s in ("sure", "guess", "unknown")}
        print(f"{name}: riders {len(res)}  sure {k['sure']}  guess {k['guess']}  unknown {k['unknown']}  "
              f"qwen failures {sum(q['error'] is not None for q in qlog)}  {total_ms:.0f} ms", flush=True)
        print("   sure:", sorted(r["no"] for r in res if r["state"] == "sure"), " guess:", sorted(r["no"] for r in res if r["state"] == "guess"), flush=True)
    try:  # release the Qwen model so the GPU is left as we found it
        body = json.dumps({"model": MODEL, "keep_alive": 0}).encode()
        urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/generate", data=body, headers={"Content-Type": "application/json"}), timeout=60).read()
    except Exception as e:
        print("could not unload the model:", repr(e))


if __name__ == "__main__":
    main()
