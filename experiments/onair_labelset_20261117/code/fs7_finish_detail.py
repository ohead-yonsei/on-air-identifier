"""POST-HOC detail of the finish-stretch frames: what the identifier labelled, left to right. usage: fs7_finish_detail.py DIR FOLDER"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
O, folder = ROOT / sys.argv[1], sys.argv[2]
man = json.loads((O / "manifest.json").read_text())
order = json.loads((O / "finish_order.json").read_text())["order"]
for m in man["frames"]:
    if m["race"] not in order:
        continue
    rs = json.loads((O / "out" / f"{m['frame']}.json").read_text())["riders"]
    line = []
    if rs:
        P = np.array(json.loads((O / folder / f"{m['frame']}.json").read_text())["probs"])
        Sc = P.copy()
        q = [r["qwen_number"] for r in rs]
        for i, n in enumerate(q):
            if n:
                Sc[i, n - 1] += 1.0
        ri, cj = linear_sum_assignment(-Sc)
        a = {int(i): int(j) for i, j in zip(ri, cj)}
        for i in sorted(a, key=lambda i: -(rs[i]["box"][0] + rs[i]["box"][2])):
            b = rs[i]["box"]
            line.append(f"{a[i] + 1}{'*' if (q[i] == a[i] + 1 or P[i, a[i]] >= 0.8) else ''}(p{P[i, a[i]]:.2f},q{q[i] or '-'},cnn{int(P[i].argmax()) + 1},x{int((b[0] + b[2]) / 2)},yb{int(b[3])},h{int(b[3] - b[1])})")
    print(m["frame"], "finish top3", order[m["race"]][:3], "| right-to-left:", "  ".join(line))
