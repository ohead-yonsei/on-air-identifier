"""Cap classifier trained to cope with small (low-resolution) caps. Same data and settings as od6_train_two_day.py
(598 hand-labelled riders), plus one augmentation: with probability 0.5 the 224-px input is shrunk to 15-70 % of its size
and enlarged again, which imitates a rider who is 40-130 px tall in the picture. See lowres/DESIGN_KO.md.
usage: lr1_train_lowres.py OUT.pt"""
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(ROOT / "model_cache")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

SEED, EPOCHS, LR, BATCH, WD = 20261117, 30, 3e-4, 32, 1e-4
P_LOW, S_MIN, S_MAX = 0.5, 0.15, 0.70
rows = []
for base in (ROOT, ROOT / "batch2", ROOT / "batch2b"):
    for r in json.loads((base / "results" / "evidence.json").read_text()):
        if r["human_number"] and not r["duplicate_of"]:
            rows.append({"y": r["human_number"] - 1, "img": Image.open(base / r["R0_35"]["input"]).convert("RGB")})
for r in json.loads((ROOT / "otherday" / "label" / "labels.json").read_text())["riders"]:
    if r["human_number"]:
        rows.append({"y": r["human_number"] - 1, "img": Image.open(ROOT / "otherday" / "label" / "inputs" / f"{r['id']}_R0_35.png").convert("RGB")})
print("training riders:", len(rows), flush=True)


def lowres(img):
    if random.random() >= P_LOW:
        return img
    s = random.uniform(S_MIN, S_MAX)
    w = max(8, int(round(img.width * s)))
    return img.resize((w, w), Image.Resampling.BILINEAR).resize((img.width, img.height), Image.Resampling.LANCZOS)


norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.Lambda(lowres), transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), norm])
seed = SEED + 399
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
torch.save(net.state_dict(), ROOT / "cnn" / "models" / sys.argv[1])
print("saved", sys.argv[1])
