"""far: the cap probabilities of another classifier for the riders already found in the dense pictures (same boxes), as h4_reprob.py,
writing into far/dense. usage: x2_reprob.py WEIGHTS.pt NAME DAY [DAY ...]
Sources per day: finish3/dense/DAY.json (whole race, 0.5 s) -> far/dense/NAME/whole/DAY.json; the pictures of the last seconds
(finish2/dense/DAY.json, else winner/dense/DAY.json) -> far/dense/NAME/last/DAY.json."""
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
    last_src = ROOT / "finish2" / "dense" / f"{day}.json"
    if not last_src.exists():
        last_src = ROOT / "winner" / "dense" / f"{day}.json"
    for kind, src in (("whole", ROOT / "finish3" / "dense" / f"{day}.json"), ("last", last_src)):
        if not src.exists():
            continue
        data = json.loads(src.read_text())
        n = 0
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
                        p["probs"] = torch.softmax(net(torch.stack(xs).cuda()), 1).cpu().numpy().round(4).tolist()
                    n += len(xs)
                i += 1
            cap.release()
        out = ROOT / "far" / "dense" / name / kind / f"{day}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data))
        print(day, kind, "from", src.parent.parent.name, "riders", n, flush=True)
