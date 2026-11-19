"""caps2, step 2: cap classifier trained on the 598 hand labels plus the mined labels (h1_mine_caps.py, h2_mine_caps_official.py).
usage: h3_train_caps.py OUT.pt [MINED_DIR ...] [--seed N] [--eps E]         writes cnn/models/OUT.pt; with no MINED_DIR: hand labels only

Same network, picture size, augmentation and schedule as the current classifier (od6_train_two_day.py: ResNet-18, 30 epochs, AdamW 3e-4,
batch 32). Hand labels are repeated three times per epoch (as wk1_train_weak.py). A mined label is taken as "this number, wrong with
probability eps" (loss -log((1-eps) P(label) + eps (1-P(label))), eps 0.1 by default): about one mined label in ten looked wrong by eye."""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(ROOT / "model_cache")
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

SEED, EPOCHS, LR, BATCH, WD, HUMAN_REPEAT = 20261117, 30, 3e-4, 32, 1e-4, 3
args = sys.argv[1:]
out_name, mined_dirs, seed_add, eps = args.pop(0), [], 199, 0.1
while args:
    a = args.pop(0)
    if a == "--seed":
        seed_add = int(args.pop(0))
    elif a == "--eps":
        eps = float(args.pop(0))
    else:
        mined_dirs.append(a)

human = []
for base in (ROOT, ROOT / "batch2", ROOT / "batch2b"):
    for r in json.loads((base / "results" / "evidence.json").read_text()):
        if r["human_number"] and not r["duplicate_of"]:
            human.append({"label": r["human_number"] - 1, "eps": 0.0, "img": Image.open(base / r["R0_35"]["input"]).convert("RGB")})
for r in json.loads((ROOT / "otherday" / "label" / "labels.json").read_text())["riders"]:
    if r["human_number"]:
        human.append({"label": r["human_number"] - 1, "eps": 0.0, "img": Image.open(ROOT / "otherday" / "label" / "inputs" / f"{r['id']}_R0_35.png").convert("RGB")})
mined = []
for d in mined_dirs:
    for r in json.loads((ROOT / d / "mined.json").read_text())["rows"]:
        mined.append({"label": r["class"] - 1, "eps": eps, "img": Image.open(ROOT / d / "crops" / r["crop"]).convert("RGB")})
rows = human * HUMAN_REPEAT + mined
print(f"hand labels {len(human)} (x{HUMAN_REPEAT}) + mined labels {len(mined)} from {mined_dirs} (eps {eps}) = {len(rows)} items per epoch; "
      f"mined by class {sorted(Counter(r['label'] + 1 for r in mined).items())}", flush=True)
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), norm])
seed = SEED + seed_add
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
        y = torch.tensor([r["label"] for r in b]).cuda()
        e = torch.tensor([r["eps"] for r in b]).cuda()
        p = torch.softmax(net(x), 1).gather(1, y[:, None])[:, 0].clamp(1e-7, 1.0)
        loss = -torch.log((1 - e) * p + e * (1 - p) + 1e-7).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss) * len(b)
    sched.step()
    if ep % 5 == 4:
        print(f"epoch {ep + 1} loss {tot / len(rows):.4f}", flush=True)
torch.save(net.state_dict(), ROOT / "cnn" / "models" / out_name)
print("saved", out_name)
