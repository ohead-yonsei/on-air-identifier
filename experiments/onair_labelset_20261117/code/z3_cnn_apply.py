"""Apply the race-excluded cap classifier to every rider box of the stripset frames (current and past).

Input crop is built exactly as for Qwen: top 35% of the rider box on a 224x224 grey canvas. Writes stripset/cnn_probs/<name>.json.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(ROOT / "model_cache")
sys.path.insert(0, str(ROOT / "code"))

import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import resnet18

from s3_classify import region

S = ROOT / "stripset"
(S / "cnn_probs").mkdir(exist_ok=True)
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
manifest = json.loads((S / "manifest.json").read_text())["frames"]
nets = {}


def net_for(race):
    if race not in nets:
        n = resnet18()
        n.fc = nn.Linear(n.fc.in_features, 11)
        n.load_state_dict(torch.load(ROOT / "cnn" / "models" / f"excl_{race}.pt", map_location="cpu"))
        nets[race] = n.eval().cuda()
    return nets[race]


count = 0
for m in manifest:
    for name, img, res in [(m["frame"], S / "frames" / f"{m['frame']}.png", S / "out" / "D0" / f"{m['frame']}.json")] + [
            (f"{m['frame']}_{t}", S / "frames_past" / f"{m['frame']}_{t}.png", S / "out_past" / "D0" / f"{m['frame']}_{t}.json") for t in ("m05", "m10", "m15")]:
        out = S / "cnn_probs" / f"{name}.json"
        if out.exists():
            continue
        riders = json.loads(res.read_text())["riders"]
        probs = []
        if riders:
            bgr = cv2.imread(str(img))
            H, W = bgr.shape[:2]
            rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            xs = []
            for r in riders:
                crop = rgb.crop(region(r["box"], "R0_35", W, H))
                can = Image.new("RGB", (224, 224), (96, 96, 96))
                patch = ImageOps.contain(crop, (224, 224), Image.Resampling.LANCZOS)
                can.paste(patch, ((224 - patch.width) // 2, (224 - patch.height) // 2))
                xs.append(tf(can))
            with torch.inference_mode():
                probs = torch.softmax(net_for(m["race"])(torch.stack(xs).cuda()), 1).cpu().numpy().round(5).tolist()
        out.write_text(json.dumps({"name": name, "race": m["race"], "probs": probs}))
        count += 1
print("written", count, "| total", len(list((S / "cnn_probs").glob("*.json"))))
