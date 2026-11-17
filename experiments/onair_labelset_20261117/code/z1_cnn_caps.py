"""Fine-tune ResNet-18 on the 417 labelled cap crops, leave one race out. See cnn/DESIGN_KO.md."""
import json
import os
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(ROOT / "model_cache")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

SEED, EPOCHS, LR, BATCH, WD, CONF = 20261117, 30, 3e-4, 32, 1e-4, 0.8
HARD = (4, 7, 10, 11)
rows = []
for batch, base in (("1", ROOT), ("2A", ROOT / "batch2"), ("2B", ROOT / "batch2b")):
    for r in json.loads((base / "results" / "evidence.json").read_text()):
        if r["human_number"] and not r["duplicate_of"]:
            rows.append({"key": f"{batch}:{r['id']}", "race": r["race"], "y": r["human_number"] - 1, "img": Image.open(base / r["R0_35"]["input"]).convert("RGB"),
                         "q9": r["R0_35"]["vlm"]["number"]})
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), norm])
test_tf = transforms.Compose([transforms.ToTensor(), norm])
races = sorted({r["race"] for r in rows})
probs = {}
for fold, race in enumerate(races):
    torch.manual_seed(SEED + fold)
    random.seed(SEED + fold)
    np.random.seed(SEED + fold)
    tr = [r for r in rows if r["race"] != race]
    te = [r for r in rows if r["race"] == race]
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
    net.eval()
    with torch.inference_mode():
        for i in range(0, len(te), 64):
            b = te[i:i + 64]
            p = torch.softmax(net(torch.stack([test_tf(r["img"]) for r in b]).cuda()), 1).cpu().numpy()
            for r, pp in zip(b, p):
                probs[r["key"]] = pp
    print(race, "train", len(tr), "test", len(te), "fold accuracy", round(float(np.mean([probs[r["key"]].argmax() == r["y"] for r in te])), 3), flush=True)


def tally(sel):
    k = Counter()
    for r in sel:
        p = probs[r["key"]]
        ok = int(p.argmax()) == r["y"]
        k["correct"] += ok
        if p.max() >= CONF:
            k["confident"] += 1
            k["confident_correct"] += ok
    n = len(sel)
    return {"n": n, "correct": k["correct"], "accuracy": k["correct"] / n, "confident": k["confident"], "confident_correct": k["confident_correct"],
            "confident_precision": k["confident_correct"] / k["confident"] if k["confident"] else None, "confident_share": k["confident"] / n}


hard = tally([r for r in rows if r["y"] + 1 in HARD])
useful = bool(hard["confident_precision"] is not None and hard["confident_precision"] >= 0.8 and hard["confident_share"] >= 0.3)
out = {"all": tally(rows), "hard_4_7_10_11": hard, "useful": useful,
       "by_number": {n: tally([r for r in rows if r["y"] + 1 == n]) for n in range(1, 12)},
       "confusion": {n: dict(Counter(int(probs[r["key"]].argmax()) + 1 for r in rows if r["y"] + 1 == n)) for n in range(1, 12)},
       "agree_with_qwen9b_when_both_answer": None,
       "settings": {"seed": SEED, "epochs": EPOCHS, "lr": LR, "batch": BATCH, "weight_decay": WD, "confident_threshold": CONF}}
both = [r for r in rows if r["q9"] != 0]
out["qwen9b_answered"] = {"n": len(both), "cnn_agrees": sum(int(probs[r["key"]].argmax()) + 1 == r["q9"] for r in both),
                          "both_agree_and_correct": sum(int(probs[r["key"]].argmax()) + 1 == r["q9"] == r["y"] + 1 for r in both),
                          "agree_total_precision_note": "both_agree_and_correct / cnn_agrees"}
(ROOT / "cnn" / "score.json").write_text(json.dumps(out, indent=1))
(ROOT / "cnn" / "probs.json").write_text(json.dumps({k: [round(float(x), 5) for x in v] for k, v in probs.items()}))
a = out["all"]
print(f"all {a['n']}: correct {a['correct']} ({a['accuracy']:.3f}) | confident (p>={CONF}) {a['confident']} of which correct {a['confident_correct']} ({a['confident_precision']:.3f})")
print(f"hard colours {hard['n']}: correct {hard['correct']} ({hard['accuracy']:.3f}) | confident {hard['confident']} ({hard['confident_share']:.3f}) of which correct {hard['confident_correct']}"
      + (f" ({hard['confident_precision']:.3f})" if hard["confident_precision"] is not None else ""))
print("by number  n / correct / confident / confident-correct")
for n in range(1, 12):
    v = out["by_number"][n]
    print(f"  {n:2d}: {v['n']:3d} / {v['correct']:3d} / {v['confident']:3d} / {v['confident_correct']:3d}")
q = out["qwen9b_answered"]
print(f"where Qwen 9B answered ({q['n']}): CNN agrees {q['cnn_agrees']}, of which correct {q['both_agree_and_correct']}")
print("useful by the fixed rule:", useful)
