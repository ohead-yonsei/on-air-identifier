"""caps2, step 3: the cap probabilities of another classifier for the riders already found in the dense pictures (same boxes, new numbers).
usage: h4_reprob.py WEIGHTS.pt NAME DAY [DAY ...]
Reads finish3/dense/DAY.json (whole race, 0.5 s) and, when present, finish2/dense/DAY.json (last 37 s, 0.2 s); writes the same files with
the "probs" replaced under caps2/dense/NAME/whole/DAY.json and caps2/dense/NAME/last/DAY.json. The cap picture is made exactly as the
identifier makes it (graphics masked, upper 35 % of the rider box, 224x224). With the current weights this reproduces the stored
probabilities (checked by h4_reprob.py cnn/models/all_two_days.pt CHECK DAY)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "identifier"))
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
from identify import MASKS
from s3_classify import region as cap_region
from torchvision import transforms
from torchvision.models import resnet18

weights, name, days = sys.argv[1], sys.argv[2], sys.argv[3:]
net = resnet18()
net.fc = nn.Linear(net.fc.in_features, 11)
net.load_state_dict(torch.load(ROOT / weights, map_location="cpu"))
net = net.eval().cuda()
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
masks = MASKS["sep06"]
for day in days:
    fc.FOLDER.setdefault(day, day)
    tms = fc.times(day)
    for kind, src in (("whole", ROOT / "finish3" / "dense" / f"{day}.json"), ("last", ROOT / "finish2" / "dense" / f"{day}.json")):
        if not src.exists():
            continue
        data = json.loads(src.read_text())
        worst, n = 0.0, 0
        for race, dr in sorted(data["races"].items()):
            want = {p["frame"]: p for p in dr["pictures"] if p["boxes"]}
            if not want:
                continue
            cap = cv2.VideoCapture(str(ROOT / fc.FOLDER[day] / "videos" / tms[race]["file"]))
            first, last = min(want), max(want)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            i = first
            while i <= last:
                ok, bgr = cap.read()
                if not ok:
                    break
                if i in want:
                    p = want[i]
                    H, W = bgr.shape[:2]
                    for x1, y1, x2, y2 in masks:
                        bgr[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = 48
                    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    xs = []
                    for box in p["boxes"]:
                        can = Image.new("RGB", (224, 224), (96, 96, 96))
                        patch = ImageOps.contain(rgb.crop(cap_region(box, "R0_35", W, H)), (224, 224), Image.Resampling.LANCZOS)
                        can.paste(patch, ((224 - patch.width) // 2, (224 - patch.height) // 2))
                        xs.append(tf(can))
                    with torch.inference_mode():
                        P = torch.softmax(net(torch.stack(xs).cuda()), 1).cpu().numpy().round(4)
                    worst = max(worst, float(np.abs(P - np.array(p["probs"])).max()))
                    n += len(xs)
                    p["probs"] = P.tolist()
                i += 1
            cap.release()
        out = ROOT / "caps2" / "dense" / name / kind / f"{day}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data))
        print(day, kind, "riders", n, "largest change of a probability against the stored ones", round(worst, 4), flush=True)
