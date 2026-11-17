"""Apply a saved cap classifier to every rider box of a day's frames. usage: wk2_apply.py WEIGHTS.pt DAY_DIR OUT_FOLDER
DAY_DIR is otherday, otherday2 or otherday3 (frames/, frames_past/, out/, out_past/, manifest.json)."""
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

weights, day, folder = sys.argv[1:4]
O = ROOT / day
man = json.loads((O / "manifest.json").read_text())
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
items = [(m["frame"], O / "frames" / f"{m['frame']}.png", O / "out" / f"{m['frame']}.json") for m in man["frames"]] + \
        [(p["name"], O / "frames_past" / f"{p['name']}.png", O / "out_past" / f"{p['name']}.json") for p in man["past"]]
net = resnet18()
net.fc = nn.Linear(net.fc.in_features, 11)
net.load_state_dict(torch.load(ROOT / "cnn" / "models" / weights, map_location="cpu"))
net = net.eval().cuda()
(O / folder).mkdir(exist_ok=True)
for name, img, res in items:
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
            probs = torch.softmax(net(torch.stack(xs).cuda()), 1).cpu().numpy().round(5).tolist()
    (O / folder / f"{name}.json").write_text(json.dumps({"name": name, "weights": weights, "probs": probs}))
print("applied", weights, "to", len(items), "frames of", day, "->", folder)
