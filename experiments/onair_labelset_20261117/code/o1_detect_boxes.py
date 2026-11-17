"""Save person and horse boxes (score >= 0.5, NMS 0.45) for every stripset frame. No Qwen calls.

Writes stripset/det/<frame>.json. See order/DESIGN_KO.md.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(ROOT / "model_cache")
sys.path.insert(0, str(ROOT / "identifier"))

import cv2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

from identify import wait_gpu
from identify2 import pick, raw_detect

out = ROOT / "stripset" / "det"
out.mkdir(exist_ok=True)
frames = json.loads((ROOT / "stripset" / "manifest.json").read_text())["frames"]
wait_gpu()
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().cuda()
for m in frames:
    p = out / f"{m['frame']}.json"
    if p.exists():
        continue
    bgr = cv2.imread(str(ROOT / "stripset" / "frames" / f"{m['frame']}.png"))
    b, s, l = raw_detect(model, bgr)
    p.write_text(json.dumps({"frame": m["frame"], "width": bgr.shape[1], "height": bgr.shape[0],
                             "persons": [[*map(float, box), float(sc)] for box, sc in pick(b, s, l, 1, 0.5)],
                             "horses": [[*map(float, box), float(sc)] for box, sc in pick(b, s, l, 19, 0.5)]}))
print("frames", len(frames), "saved", len(list(out.glob("*.json"))))
