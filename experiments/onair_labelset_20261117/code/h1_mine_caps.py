"""caps2, step 1: find riders whose number the cap classifier does not read, and label them from the running-order strip. No hand labelling.
usage: h1_mine_caps.py OUT_DIR [--probs FOLDER] DAY [DAY ...]         DAY has da/DAY/{manifest.json, frames, frames_past, out, out_past, cnn_probs_two}
       --probs FOLDER (round 2): the folder under da/DAY with another classifier's probabilities (wk2_apply.py)

Idea. The strip gives the order of the first five horses at that moment. In a picture, the riders the classifier reads with confidence and
whose numbers are on the strip ("anchors") show where the strip's order lies in the picture. A strip number with no anchor must be the
rider standing at its place in that order. That rider gets the strip number as its label: an example the classifier got wrong or was unsure of.

Rules (all must hold; chosen for precision, not yield):
  - the strip shows five different numbers, at most one of them 11 or more (11+ share one cap class), and its first four are the same as
    on the time point 3 s before (the order is not changing)
  - exactly one of the strip's first four numbers (the target) has no anchor; the other three have one. Anchor = the classifier's assigned
    number (one-to-one over the picture) with probability 0.9 or more
  - the three anchors lie in strip order along their main direction in the picture, each step at least 0.3 rider heights
  - exactly one other rider lies at the target's place: ahead of the second for a target in first place (by 0.3 heights or more), between its
    two neighbours otherwise (0.2 heights from each); for a target in fourth place the fifth must be an anchor too
  - the rider at that place is at least 40 px tall
Writes OUT_DIR/mined.json (one row per label, with the classifier's own reading) and OUT_DIR/crops/NAME_k.png (the 224x224 cap picture the
classifier sees). Nothing here is used by the identifier at run time; the strip is used only to make training labels."""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from s3_classify import region

ANCHOR_P, STEP, GAP, MIN_H = 0.9, 0.3, 0.2, 40


def cls(n):
    return min(n, 11)


def canvas(rgb, box, W, H):
    crop = rgb.crop(region(box, "R0_35", W, H))
    can = Image.new("RGB", (224, 224), (96, 96, 96))
    patch = ImageOps.contain(crop, (224, 224), Image.Resampling.LANCZOS)
    can.paste(patch, ((224 - patch.width) // 2, (224 - patch.height) // 2))
    return can


def mine_picture(boxes, P, strip):
    """-> (index of the rider, target number, target place 1-4, info) or None"""
    n = len(boxes)
    if n < 4:
        return None
    ri, cj = linear_sum_assignment(-P)
    num = {int(i): int(j) + 1 for i, j in zip(ri, cj)}
    conf = {i: float(P[i, num[i] - 1]) for i in num}
    c5 = [cls(s) for s in strip]
    anchor = {}
    for k, c in enumerate(c5):
        got = [i for i in num if num[i] == c and conf[i] >= ANCHOR_P]
        if got:
            anchor[k] = got[0]
    missing = [k for k in range(4) if k not in anchor]
    if len(missing) != 1:
        return None
    k = missing[0]
    if k == 3 and 4 not in anchor:
        return None
    B = np.array(boxes, float)
    pos = np.c_[(B[:, 0] + B[:, 2]) / 2, B[:, 3]]
    h = B[:, 3] - B[:, 1]
    ks = sorted(a for a in anchor if a < 4 or k == 3)
    A = pos[[anchor[a] for a in ks]]
    d = np.linalg.svd(A - A.mean(0))[2][0]                 # main direction of the anchors
    proj = pos @ d
    pa = {a: proj[anchor[a]] for a in ks}
    if pa[ks[0]] < pa[ks[-1]]:                              # orient: larger = further ahead
        proj, pa = -proj, {a: -v for a, v in pa.items()}
    hm = float(np.mean(h[[anchor[a] for a in ks]]))
    if any(pa[ks[j]] - pa[ks[j + 1]] < STEP * hm for j in range(len(ks) - 1)):
        return None
    used = set(anchor.values())
    cand = [i for i in range(n) if i not in used]
    if k == 0:
        here = [i for i in cand if proj[i] - pa[1] >= STEP * hm]
    else:
        here = [i for i in cand if pa[k - 1] - proj[i] >= GAP * hm and proj[i] - pa[k + 1] >= GAP * hm]
    if len(here) != 1 or h[here[0]] < MIN_H:
        return None
    i = here[0]
    return i, strip[k], k + 1, {"read_as": num.get(i), "read_p": round(conf.get(i, 0.0), 4), "p_target": round(float(P[i, cls(strip[k]) - 1]), 4), "riders": n, "height": round(float(h[i]), 1)}


def main():
    args = sys.argv[1:]
    out, folder = ROOT / args.pop(0), "cnn_probs_two"
    if args[0] == "--probs":
        args.pop(0)
        folder = args.pop(0)
    days = args
    (out / "crops").mkdir(parents=True, exist_ok=True)
    rows, seen = [], Counter()
    for day in days:
        O = ROOT / "da" / day
        man = json.loads((O / "manifest.json").read_text())
        frames = man["frames"]
        prev = {}
        for m in frames:
            race, t = m["frame"].split("_t")
            prev[(race, int(t))] = m["strip"]
        for m in frames:
            S = m["strip"]
            race, t = m["frame"].split("_t")
            before = prev.get((race, int(t) - 3))
            if len(S) < 5 or len(set(S)) != 5 or sum(s >= 11 for s in S) > 1 or before is None or before[:4] != S[:4]:
                continue
            for name, img, res in ((m["frame"], O / "frames" / f"{m['frame']}.png", O / "out" / f"{m['frame']}.json"),
                                   (f"{m['frame']}_m05", O / "frames_past" / f"{m['frame']}_m05.png", O / "out_past" / f"{m['frame']}_m05.json"),
                                   (f"{m['frame']}_m10", O / "frames_past" / f"{m['frame']}_m10.png", O / "out_past" / f"{m['frame']}_m10.json")):
                seen["pictures"] += 1
                riders = json.loads(res.read_text())["riders"]
                if len(riders) < 4:
                    continue
                P = np.array(json.loads((O / folder / f"{name}.json").read_text())["probs"])
                boxes = [r["box"] for r in riders]
                got = mine_picture(boxes, P, S)
                if got is None:
                    continue
                i, number, place, info = got
                bgr = cv2.imread(str(img))
                H, W = bgr.shape[:2]
                crop = f"{day}_{name}_{i}.png"
                canvas(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), boxes[i], W, H).save(out / "crops" / crop)
                rows.append({"day": day, "picture": name, "rider": i, "box": boxes[i], "number": number, "class": cls(number), "place": place, "strip": S, "crop": crop, **info})
        print(day, "labels so far", len(rows), flush=True)
    by = Counter(r["class"] for r in rows)
    wrong = Counter(r["class"] for r in rows if r["read_as"] != r["class"] and r["read_p"] >= 0.7)
    print("pictures looked at", seen["pictures"], "labels", len(rows))
    print("by class", sorted(by.items()))
    print("of which the classifier was sure of another number", sorted(wrong.items()))
    (out / "mined.json").write_text(json.dumps({"days": days, "rules": {"anchor_p": ANCHOR_P, "step": STEP, "gap": GAP, "min_height": MIN_H}, "by_class": dict(sorted(by.items())),
                                                "sure_of_another_number": dict(sorted(wrong.items())), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
