"""Identifier with selectable rider-detection variants (D0-D3). See stripset/DESIGN_KO.md.

Everything after detection is the same as identify.py (same features, Qwen protocol, assignment, sure rule).

usage: python identify2.py FRAME.png [...] --out OUTDIR --variant D2 --model MODEL.json [--masks sep06] [--no-overlay]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.ops import nms

from identify import COLOUR, ENDPOINT, MASKS, MODEL, W_QWEN, ColourModel, ask_qwen, draw, features, region, wait_gpu

VARIANTS = {  # person score threshold, min rider height at 1920 width, zoom pass
    "D0": (0.5, 70, False), "D1": (0.3, 40, False), "D2": (0.5, 40, True), "D3": (0.3, 40, True),
}
LOW = 0.3  # score used to find where the horses are for the zoom pass


def raw_detect(model, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().cuda() / 255
    with torch.inference_mode():
        p = model([t])[0]
    return p["boxes"].cpu().numpy(), p["scores"].cpu().numpy(), p["labels"].cpu().numpy()


def pick(boxes, scores, labels, cls, thr):
    m = (labels == cls) & (scores >= thr)
    if not m.any():
        return []
    b, s = torch.from_numpy(boxes[m]).float(), torch.from_numpy(scores[m]).float()  # zoomed boxes come back as float64
    keep = nms(b, s, 0.45)
    return list(zip(b[keep].tolist(), s[keep].tolist()))


def find_riders(model, bgr, masks, variant):
    thr, min_h1920, zoom = VARIANTS[variant]
    H, W = bgr.shape[:2]
    boxes, scores, labels = raw_detect(model, bgr)
    if zoom:
        m = (scores >= LOW) & np.isin(labels, (1, 19))
        if m.any():
            x1, y1 = boxes[m][:, 0].min(), boxes[m][:, 1].min()
            x2, y2 = boxes[m][:, 2].max(), boxes[m][:, 3].max()
            px, py = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
            x1, y1, x2, y2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(W, x2 + px)), int(min(H, y2 + py))
            crop = bgr[y1:y2, x1:x2]
            if crop.shape[0] >= 32 and crop.shape[1] >= 32:
                f = min(2.0, 2600 / max(crop.shape[:2]))
                big = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
                zb, zs, zl = raw_detect(model, big)
                zb = zb / f + np.array([x1, y1, x1, y1])
                boxes, scores, labels = np.concatenate([boxes, zb]), np.concatenate([scores, zs]), np.concatenate([labels, zl])
    persons, horses = pick(boxes, scores, labels, 1, thr), pick(boxes, scores, labels, 19, 0.5)
    inter = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda a: max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    min_h = min_h1920 * W / 1920
    kept = []
    for box, score in persons:
        l, t, r, b = box
        if b - t < min_h or l < 4 or t < 4 or r > W - 4 or b > H - 4:
            continue
        cx, cy = (l + r) / 2, (t + b) / 2
        if any(mx1 * W <= cx <= mx2 * W and my1 * H <= cy <= my2 * H for mx1, my1, mx2, my2 in masks):
            continue
        if not any(inter(box, hb) / area(box) >= 0.15 for hb, _ in horses):
            continue
        kept.append((box, score))
    out = [(box, score) for box, score in kept
           if not any(o is not box and area(o) > area(box) and inter(box, o) / area(box) > 0.7 for o, _ in kept)]
    return sorted(out, key=lambda x: (x[0][0] + x[0][2]) / 2)


def identify(bgr, det, cm, masks, variant, qlog):
    H, W = bgr.shape[:2]
    for x1, y1, x2, y2 in masks:
        bgr[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = 48
    riders = find_riders(det, bgr, masks, variant)
    if not riders:
        return []
    P, Q = [], []
    for box, _ in riders:
        P.append(cm.proba(features(bgr, region(box, "R0_27", W, H), box)))
        Q.append(ask_qwen(bgr, region(box, "R0_35", W, H), qlog))
    P = np.array(P)
    S = P.copy()
    for i, q in enumerate(Q):
        if q:
            S[i, q - 1] += W_QWEN
    ri, cj = linear_sum_assignment(-S)  # with more than 11 riders the weakest ones stay unassigned
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
                    "p_learned": round(float(P[i, j]), 4) if j is not None else None, "detector_score": round(det_score, 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--masks", default="sep06", choices=list(MASKS))
    ap.add_argument("--no-overlay", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wait_gpu()
    det = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().cuda()
    cm = ColourModel(a.model)
    for f in a.frames:
        name = Path(f).stem
        if (out / f"{name}.json").exists():
            continue  # resumable
        bgr = cv2.imread(f)
        qlog, start = [], time.perf_counter()
        res = identify(bgr, det, cm, MASKS[a.masks], a.variant, qlog)
        total_ms = (time.perf_counter() - start) * 1000
        (out / f"{name}.json").write_text(json.dumps({"frame": name, "variant": a.variant, "model": a.model, "riders": res,
                                                      "qwen_failures": sum(q["error"] is not None for q in qlog), "total_ms": round(total_ms, 1)}))
        if not a.no_overlay:
            draw(bgr, res, out / f"{name}_overlay.jpg")
    print(a.variant, Path(a.model).stem, "frames", len(a.frames), flush=True)
    try:
        urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/generate", data=json.dumps({"model": MODEL, "keep_alive": 0}).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=60).read()
    except Exception as e:
        print("could not unload the model:", repr(e))


if __name__ == "__main__":
    main()
