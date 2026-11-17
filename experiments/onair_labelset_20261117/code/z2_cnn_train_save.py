"""Train and SAVE the cap classifier: one model per held-out race, plus one on all riders. Same settings as z1_cnn_caps.py."""
import json
import os
import random
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
rows = []
for base in (ROOT, ROOT / "batch2", ROOT / "batch2b"):
    for r in json.loads((base / "results" / "evidence.json").read_text()):
        if r["human_number"] and not r["duplicate_of"]:
            rows.append({"race": r["race"], "y": r["human_number"] - 1, "img": Image.open(base / r["R0_35"]["input"]).convert("RGB")})
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), norm])
out = ROOT / "cnn" / "models"
out.mkdir(parents=True, exist_ok=True)
races = sorted({r["race"] for r in rows})
for fold, race in enumerate(races + [None]):
    seed = SEED + (fold if race is not None else 99)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tr = [r for r in rows if r["race"] != race]
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, 11)
    net = net.cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    for ep in range(EPOCHS):
        net.train()
        order = list(range(len(tr)))
        random.shuffle(order)
        for i in range(0, len(order), BATCH):
            b = [tr[j] for j in order[i:i + BATCH]]
            if len(b) < 2:
                continue
            x = torch.stack([train_tf(r["img"]) for r in b]).cuda()
            y = torch.tensor([r["y"] for r in b]).cuda()
            loss = nn.functional.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    name = f"excl_{race}.pt" if race else "all.pt"
    torch.save(net.state_dict(), out / name)
    print(name, "trained on", len(tr), flush=True)
