"""Train the cap classifier on two race days: 2026-09-06 (417 riders) + 2026-11-14 (labelled riders). See otherday2/DESIGN_KO.md.

Same settings as z2_cnn_train_save.py. Writes cnn/models/all_two_days.pt and otherday/label/inputs/*.png.
"""
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(ROOT / "model_cache")
sys.path.insert(0, str(ROOT / "code"))

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from s3_classify import region

SEED, EPOCHS, LR, BATCH, WD = 20261117, 30, 3e-4, 32, 1e-4
rows = []
for base in (ROOT, ROOT / "batch2", ROOT / "batch2b"):
    for r in json.loads((base / "results" / "evidence.json").read_text()):
        if r["human_number"] and not r["duplicate_of"]:
            rows.append({"y": r["human_number"] - 1, "img": Image.open(base / r["R0_35"]["input"]).convert("RGB")})
n_day1 = len(rows)
O = ROOT / "otherday"
(O / "label" / "inputs").mkdir(exist_ok=True)
cache = {}
for r in json.loads((O / "label" / "labels.json").read_text())["riders"]:
    if not r["human_number"]:
        continue
    if r["frame"] not in cache:
        cache[r["frame"]] = cv2.imread(str(O / "frames" / f"{r['frame']}.png"))
    bgr = cache[r["frame"]]
    H, W = bgr.shape[:2]
    crop = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).crop(region(r["person_box"], "R0_35", W, H))
    can = Image.new("RGB", (224, 224), (96, 96, 96))
    patch = ImageOps.contain(crop, (224, 224), Image.Resampling.LANCZOS)
    can.paste(patch, ((224 - patch.width) // 2, (224 - patch.height) // 2))
    can.save(O / "label" / "inputs" / f"{r['id']}_R0_35.png")
    rows.append({"y": r["human_number"] - 1, "img": can})
print("training riders:", n_day1, "from 2026-09-06 +", len(rows) - n_day1, "from 2026-11-14 =", len(rows), flush=True)
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), norm])
seed = SEED + 199
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
net.fc = nn.Linear(net.fc.in_features, 11)
net = net.cuda()
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
for ep in range(EPOCHS):
    net.train()
    order = list(range(len(rows)))
    random.shuffle(order)
    for i in range(0, len(order), BATCH):
        b = [rows[j] for j in order[i:i + BATCH]]
        if len(b) < 2:
            continue
        x = torch.stack([train_tf(r["img"]) for r in b]).cuda()
        y = torch.tensor([r["y"] for r in b]).cuda()
        loss = nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    sched.step()
torch.save(net.state_dict(), ROOT / "cnn" / "models" / "all_two_days.pt")
print("saved all_two_days.pt")
