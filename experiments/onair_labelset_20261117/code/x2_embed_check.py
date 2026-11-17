"""Do pretrained appearance embeddings tell horses apart within a race? Development races only (exploration).

ImageNet ResNet-50 pooled features of each rider crop (cap-sure riders), cosine distance, same-number vs different-number pairs
in different frames of the same race. Compare with the colour-histogram silks features (AUC 0.651).
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(ROOT / "model_cache")

import cv2
import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50

S = ROOT / "stripset"
manifest = [m for m in json.loads((S / "manifest.json").read_text())["frames"] if m["split"] == "dev"]
items = []
for m in manifest:
    items.append((m, m["frame"], S / "frames" / f"{m['frame']}.png", S / "out" / "D0" / f"{m['frame']}.json"))
    for tag in ("m05", "m10", "m15"):
        n = f"{m['frame']}_{tag}"
        items.append((m, n, S / "frames_past" / f"{n}.png", S / "out_past" / "D0" / f"{n}.json"))
weights = ResNet50_Weights.IMAGENET1K_V2
net = resnet50(weights=weights)
net.fc = torch.nn.Identity()
net = net.eval().cuda()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


def embed(crops):
    x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().cuda() / 255
    with torch.inference_mode():
        f = net((x - mean) / std)
    return torch.nn.functional.normalize(f, dim=1).cpu().numpy()


for region in ("whole", "torso"):
    by_race = {}
    for m, name, img, res in items:
        riders = [r for r in json.loads(res.read_text())["riders"] if r["state"] == "sure"]
        if not riders:
            continue
        bgr = cv2.imread(str(img))
        H, W = bgr.shape[:2]
        crops = []
        for r in riders:
            l, t, rr, b = r["box"]
            if region == "torso":
                t, b = t + 0.30 * (b - t), t + 0.80 * (b - t)
            l, t, rr, b = max(0, int(l)), max(0, int(t)), min(W, int(rr) + 1), min(H, int(b) + 1)
            c = cv2.cvtColor(bgr[t:b, l:rr], cv2.COLOR_BGR2RGB)
            crops.append(cv2.resize(c, (128, 192), interpolation=cv2.INTER_CUBIC))
        for r, e in zip(riders, embed(crops)):
            by_race.setdefault(m["race"], []).append((r["no"], e, name))
    rng = np.random.default_rng(0)
    same, diff = [], []
    for race, v in by_race.items():
        idx = rng.permutation(len(v))[:400]
        E = np.stack([v[i][1] for i in idx])
        D = 1 - E @ E.T
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if v[idx[a]][2] == v[idx[b]][2]:
                    continue
                (same if v[idx[a]][0] == v[idx[b]][0] else diff).append(D[a, b])
    same, diff = np.array(same), np.array(diff)
    ranks = np.argsort(np.argsort(np.concatenate([same, diff])))
    auc = 1 - (ranks[: len(same)].sum() - len(same) * (len(same) - 1) / 2) / (len(same) * len(diff))
    print(f"{region:5s}: instances {sum(len(v) for v in by_race.values())} | same pairs {len(same)} median {np.median(same):.3f} | different pairs {len(diff)} median {np.median(diff):.3f} | AUC {auc:.3f}")
print("weights:", weights, "| cache:", os.environ["TORCH_HOME"])
