"""Why does B stall? Development races only (exploration), new identifier C2.

For frames with no correct statement under (W3, K2, m2): where are the strip's top-three horses?
usage: python z5_b_diagnosis.py dev
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "stripset"
split = sys.argv[1]
frames = [m for m in json.loads((S / "manifest.json").read_text())["frames"] if m["split"] == split]
TAGS = [None, "m05", "m10"]


def labelled(frame, tag):
    name = frame if tag is None else f"{frame}_{tag}"
    riders = json.loads(((S / "out" / "D0") if tag is None else (S / "out_past" / "D0")).joinpath(f"{name}.json").read_text())["riders"]
    P = np.array(json.loads((S / "cnn_probs" / f"{name}.json").read_text())["probs"]) if riders else np.zeros((0, 11))
    if not riders:
        return []
    Sc = P.copy()
    for i, r in enumerate(riders):
        if r["qwen_number"]:
            Sc[i, r["qwen_number"] - 1] += 1.0
    ri, cj = linear_sum_assignment(-Sc)
    a = {int(i): int(j) for i, j in zip(ri, cj)}
    out = []
    for i, r in enumerate(riders):
        j = a.get(i)
        no = 0 if j is None else j + 1
        sure = j is not None and (r["qwen_number"] == no or P[i, j] >= 0.8)
        out.append({"y": r["box"][3], "no": no, "sure": bool(sure), "p": float(P[i, j]) if j is not None else 0.0})
    return sorted(out, key=lambda x: -x["y"])


k = Counter()
rank_of_missing = Counter()
riders_n = []
for m in frames:
    B3 = m["strip"][:3]
    per = [labelled(m["frame"], t) for t in TAGS]
    front_votes, any_sure_votes, any_assigned_votes = Counter(), Counter(), Counter()
    for lab in per:
        for n in {x["no"] for x in lab[:2] if x["sure"] and x["no"]}:
            front_votes[n] += 1
        for n in {x["no"] for x in lab if x["sure"] and x["no"]}:
            any_sure_votes[n] += 1
        for n in {x["no"] for x in lab if x["no"]}:
            any_assigned_votes[n] += 1
    said = {n for n, v in front_votes.items() if v >= 2}
    ok = bool(said & set(B3))
    k["frames"] += 1
    k["ok"] += ok
    riders_n.append(len(per[0]))
    k["frames_any_sure_top3 (sure x2 anywhere)"] += any(any_sure_votes[n] >= 2 for n in B3)
    if not ok:
        k["missing"] += 1
        if any(any_sure_votes[n] >= 2 for n in B3):
            k["missing: a top-3 horse is sure somewhere, but not among the front 2 boxes"] += 1
            for lab in per[:1]:
                for pos, x in enumerate(lab):
                    if x["sure"] and x["no"] in B3:
                        rank_of_missing[min(pos + 1, 6)] += 1
        elif any(any_assigned_votes[n] >= 2 for n in B3):
            k["missing: a top-3 horse has its number but is not sure"] += 1
        else:
            k["missing: no top-3 horse is given its number anywhere"] += 1
        k["missing: frames with fewer than 4 rider boxes"] += len(per[0]) < 4
print(f"split {split}: {k['frames']} frames | correct statement {k['ok']} ({k['ok'] / k['frames']:.3f}) | median rider boxes per frame {sorted(riders_n)[len(riders_n) // 2]}")
print(f"  upper bound if every sure number anywhere were said: {k['frames_any_sure_top3 (sure x2 anywhere)']} ({k['frames_any_sure_top3 (sure x2 anywhere)'] / k['frames']:.3f})")
for key, v in k.items():
    if key.startswith("missing"):
        print(f"  {key}: {v}")
print("  position (by the vertical rule) of sure top-3 horses that were outside the front 2, current frame:", dict(sorted(rank_of_missing.items())))
