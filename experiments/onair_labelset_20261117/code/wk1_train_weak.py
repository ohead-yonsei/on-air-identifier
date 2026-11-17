"""Cap classifier trained on hand labels plus strip-based partial labels. See weak/DESIGN_KO.md.

usage: wk1_train_weak.py OUT.pt DAY [DAY ...]      DAY in sep06 nov14 nov07 (days whose strip frames give partial labels)

Hand-labelled riders (598) get the usual one-number target. A front box of a strip frame gets a partial target:
"one of the numbers on the strip". Loss = -log((1-e) * P(allowed) + e * P(not allowed)), e = 0 for hand labels, 0.1 for strip labels.
Everything else is as in od6_train_two_day.py."""
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
FRONT, EPS, HUMAN_REPEAT = 3, 0.1, 3
DAYS = {"sep06": (ROOT / "stripset", "out/D0", "out_past/D0"), "nov14": (ROOT / "otherday", "out", "out_past"), "nov07": (ROOT / "otherday2", "out", "out_past")}
out_name, days = sys.argv[1], sys.argv[2:]


def canvas(rgb, box, W, H):
    crop = rgb.crop(region(box, "R0_35", W, H))
    can = Image.new("RGB", (224, 224), (96, 96, 96))
    patch = ImageOps.contain(crop, (224, 224), Image.Resampling.LANCZOS)
    can.paste(patch, ((224 - patch.width) // 2, (224 - patch.height) // 2))
    return can


human = []
for base in (ROOT, ROOT / "batch2", ROOT / "batch2b"):
    for r in json.loads((base / "results" / "evidence.json").read_text()):
        if r["human_number"] and not r["duplicate_of"]:
            human.append({"allowed": [r["human_number"] - 1], "eps": 0.0, "img": Image.open(base / r["R0_35"]["input"]).convert("RGB")})
for r in json.loads((ROOT / "otherday" / "label" / "labels.json").read_text())["riders"]:
    if r["human_number"]:
        human.append({"allowed": [r["human_number"] - 1], "eps": 0.0, "img": Image.open(ROOT / "otherday" / "label" / "inputs" / f"{r['id']}_R0_35.png").convert("RGB")})
weak, skipped = [], 0
for d in days:
    O, out, out_past = DAYS[d]
    for m in json.loads((O / "manifest.json").read_text())["frames"]:
        S = m["strip"]
        if len(S) < 3 or any(n > 11 or n < 1 for n in S):
            skipped += 1
            continue
        for name, img, res in ((m["frame"], O / "frames" / f"{m['frame']}.png", O / out / f"{m['frame']}.json"),
                               (f"{m['frame']}_m05", O / "frames_past" / f"{m['frame']}_m05.png", O / out_past / f"{m['frame']}_m05.json"),
                               (f"{m['frame']}_m10", O / "frames_past" / f"{m['frame']}_m10.png", O / out_past / f"{m['frame']}_m10.json")):
            riders = json.loads(res.read_text())["riders"]
            if not riders:
                continue
            bgr = cv2.imread(str(img))
            H, W = bgr.shape[:2]
            rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            for r in sorted(riders, key=lambda r: -r["box"][3])[:FRONT]:
                weak.append({"allowed": [n - 1 for n in S], "eps": EPS, "img": canvas(rgb, r["box"], W, H)})
rows = human * HUMAN_REPEAT + weak
print(f"hand-labelled riders {len(human)} (x{HUMAN_REPEAT}) + strip-labelled boxes {len(weak)} from {days} (frames skipped {skipped}) = {len(rows)} items per epoch", flush=True)
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), norm])
seed = SEED + 299
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
    tot = 0.0
    for i in range(0, len(order), BATCH):
        b = [rows[j] for j in order[i:i + BATCH]]
        if len(b) < 2:
            continue
        x = torch.stack([train_tf(r["img"]) for r in b]).cuda()
        mask = torch.zeros(len(b), 11)
        for k, r in enumerate(b):
            mask[k, r["allowed"]] = 1.0
        mask = mask.cuda()
        eps = torch.tensor([r["eps"] for r in b]).cuda()
        p_in = (torch.softmax(net(x), 1) * mask).sum(1).clamp(1e-7, 1.0)
        loss = -torch.log((1 - eps) * p_in + eps * (1 - p_in) + 1e-7).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss) * len(b)
    sched.step()
    if ep % 5 == 4:
        print(f"epoch {ep + 1} loss {tot / len(rows):.4f}", flush=True)
torch.save(net.state_dict(), ROOT / "cnn" / "models" / out_name)
print("saved", out_name)
