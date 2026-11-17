"""Other-day test, step 2: run the identifier (detection D0 + Qwen 9B + all-data models) and the new cap classifier on every frame."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "otherday"
os.environ["TORCH_HOME"] = str(ROOT / "model_cache")
sys.path.insert(0, str(ROOT / "code"))
man = json.loads((O / "manifest.json").read_text())
jobs = [(O / "out", [str(O / "frames" / f"{m['frame']}.png") for m in man["frames"]]),
        (O / "out_past", [str(O / "frames_past" / f"{p['name']}.png") for p in man["past"]])]
for out, files in jobs:
    for k in range(0, len(files), 60):
        subprocess.run([sys.executable, str(ROOT / "identifier" / "identify2.py"), *files[k:k + 60], "--out", str(out), "--variant", "D0",
                        "--model", str(ROOT / "identifier" / "model.json"), "--no-overlay"], check=True)

import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import resnet18

from s3_classify import region

net = resnet18()
net.fc = nn.Linear(net.fc.in_features, 11)
net.load_state_dict(torch.load(ROOT / "cnn" / "models" / "all.pt", map_location="cpu"))
net = net.eval().cuda()
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
(O / "cnn_probs").mkdir(exist_ok=True)
items = [(m["frame"], O / "frames" / f"{m['frame']}.png", O / "out" / f"{m['frame']}.json") for m in man["frames"]] + \
        [(p["name"], O / "frames_past" / f"{p['name']}.png", O / "out_past" / f"{p['name']}.json") for p in man["past"]]
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
    (O / "cnn_probs" / f"{name}.json").write_text(json.dumps({"name": name, "probs": probs}))
print("identified", len(items), "frames")
