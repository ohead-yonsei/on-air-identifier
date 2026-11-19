"""caps2, step 1b: label riders the cap classifier does not read, from the official order of the WHOLE field. No hand labelling.
usage: h2_mine_caps_official.py OUT_DIR [--source NAME] DAY [DAY ...]      needs finish3/dense/DAY.json (g4_dense_whole.py: every 0.5 s, whole race)
       --source NAME (round 2): the cap probabilities of another classifier, caps2/dense/NAME/{whole,last}/DAY.json (h4_reprob.py)
                                                           and, when present, finish2/dense/DAY.json (last 37 s, every 0.2 s)

Idea (as h1_mine_caps.py, with a richer answer). The official passing times give the order of every horse at every moment. Riders the
classifier reads with confidence ("anchors", probability 0.9 or more, a number that runs in the race) show where that order lies in the picture.
If the official order says exactly one horse runs between two anchors (or ahead of the first anchor), and exactly one other rider stands
there in the picture, that rider gets that horse's number as its label.

Rules (for precision):
  - race whose clock agrees with the official time; at least 3 anchors; anchors of the 11+ class are not used when two or more horses of
    the race have numbers of 11 or more
  - from the anchors a chain is taken in which each is at least 3 m behind the one before in the official order (at least 3 in the chain);
    the chain must lie in that order along its main direction in the picture, each step at least 0.3 rider heights. Anchors left out of
    the chain are known riders: they are neither labelled nor counted as "another rider"
  - the horse to label is the only one officially between the two anchors (or ahead of the first), and every other horse is at least
    3 m outside that stretch; the rider is the only non-anchor rider standing in that stretch (0.2 heights from each anchor; 0.3 ahead of
    the first), and at least 40 px tall
Per race and number at most 12 labels are kept (evenly over time), since neighbouring pictures show the same cap.
Writes OUT_DIR/mined.json and OUT_DIR/crops/*.png (the 224x224 cap picture the classifier sees). Training labels only; the identifier
never reads official data at run time."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
from h1_mine_caps import canvas, cls

ANCHOR_P, STEP, GAP, MIN_H, CLEAR_M, PER = 0.9, 0.3, 0.2, 40, 3.0, 12


def mine_picture(boxes, P, togo):
    """togo: number -> metres to go (official). -> list of (rider index, number, where, info)"""
    n = len(boxes)
    if n < 4:
        return []
    ri, cj = linear_sum_assignment(-P)
    num = {int(i): int(j) + 1 for i, j in zip(ri, cj)}
    conf = {i: float(P[i, num[i] - 1]) for i in num}
    field = sorted(togo, key=togo.get)                      # first = leading
    many11 = sum(h >= 11 for h in field) > 1
    by_class = defaultdict(list)
    for h in field:
        by_class[cls(h)].append(h)
    anchors = []                                            # (horse number, rider index)
    for i in num:
        c = num[i]
        if conf[i] >= ANCHOR_P and len(by_class.get(c, [])) == 1 and not (c == 11 and many11):
            anchors.append((by_class[c][0], i))
    if len(anchors) < 3:
        return []
    anchors.sort(key=lambda a: togo[a[0]])
    every = anchors                                         # all anchors: their riders are known, they are never candidates
    chain = [anchors[0]]                                    # anchors at least 3 m apart from the one before: only these carry the order
    for a in anchors[1:]:
        if togo[a[0]] - togo[chain[-1][0]] >= CLEAR_M:
            chain.append(a)
    anchors = chain
    if len(anchors) < 3:
        return []
    B = np.array(boxes, float)
    pos = np.c_[(B[:, 0] + B[:, 2]) / 2, B[:, 3]]
    h = B[:, 3] - B[:, 1]
    A = pos[[i for _, i in anchors]]
    d = np.linalg.svd(A - A.mean(0))[2][0]
    proj = pos @ d
    pa = [proj[i] for _, i in anchors]
    if pa[0] < pa[-1]:
        proj, pa = -proj, [-v for v in pa]
    hm = float(np.mean(h[[i for _, i in anchors]]))
    if any(pa[j] - pa[j + 1] < STEP * hm for j in range(len(pa) - 1)):
        return []
    used = {i for _, i in every}
    cand = [i for i in range(n) if i not in used]
    names = {a for a, _ in every}
    out = []
    spans = [(None, 0)] + [(j, j + 1) for j in range(len(anchors) - 1)]
    for a, b in spans:
        lo = -1e9 if a is None else togo[anchors[a][0]]
        hi = togo[anchors[b][0]]
        inside = [x for x in field if x not in names and lo < togo[x] < hi]
        if len(inside) != 1:
            continue
        x = inside[0]
        if any(lo - CLEAR_M < togo[y] < hi + CLEAR_M for y in field if y != x and y not in (anchors[b][0], None if a is None else anchors[a][0])):
            continue
        if a is None:
            here = [i for i in cand if proj[i] - pa[0] >= STEP * hm]
        else:
            here = [i for i in cand if pa[a] - proj[i] >= GAP * hm and proj[i] - pa[b] >= GAP * hm]
        if len(here) != 1 or h[here[0]] < MIN_H:
            continue
        i = here[0]
        out.append((i, x, "leading" if a is None else "between", {"read_as": num.get(i), "read_p": round(conf.get(i, 0.0), 4), "p_target": round(float(P[i, cls(x) - 1]), 4),
                                                                  "riders": n, "anchors": len(anchors), "height": round(float(h[i]), 1)}))
    return out


def main():
    args = sys.argv[1:]
    out, source = ROOT / args.pop(0), None
    if args[0] == "--source":
        args.pop(0)
        source = args.pop(0)
    days = args
    (out / "crops").mkdir(parents=True, exist_ok=True)
    rows = []
    for day in days:
        fc.FOLDER.setdefault(day, day)
        off, tms = fc.official(day), fc.times(day)
        sources = [ROOT / "finish3" / "dense" / f"{day}.json", ROOT / "finish2" / "dense" / f"{day}.json"]
        if source:
            sources = [ROOT / "caps2" / "dense" / source / "whole" / f"{day}.json", ROOT / "caps2" / "dense" / source / "last" / f"{day}.json"]
        found = defaultdict(list)                           # race -> rows
        for src in sources:
            if not src.exists():
                continue
            for race, dr in json.loads(src.read_text())["races"].items():
                if not dr["clock_agrees"]:
                    continue
                for p in dr["pictures"]:
                    if len(p["boxes"]) < 4:
                        continue
                    ref = fc.reference_full(off[race], dr["winner_time"] - p["to_finish"])
                    if ref is None:
                        continue
                    for i, number, where, info in mine_picture(p["boxes"], np.array(p["probs"], float), ref["to_go"]):
                        found[race].append({"day": day, "race": race, "frame": p["frame"], "to_finish": p["to_finish"], "rider": i, "box": p["boxes"][i], "number": number,
                                            "class": cls(number), "where": where, **info})
        kept = 0
        for race, rs in sorted(found.items()):
            per = defaultdict(list)
            for r in sorted(rs, key=lambda r: r["frame"]):
                if not per[r["number"]] or per[r["number"]][-1]["frame"] != r["frame"]:
                    per[r["number"]].append(r)
            keep = []
            for number, v in per.items():
                idx = sorted(set(int(round(k * (len(v) - 1) / (PER - 1))) for k in range(PER))) if len(v) > PER else range(len(v))
                keep += [v[k] for k in idx]
            want = defaultdict(list)
            for r in keep:
                want[r["frame"]].append(r)
            path = ROOT / fc.FOLDER[day] / "videos" / tms[race]["file"]
            cap = cv2.VideoCapture(str(path))
            first, last = min(want), max(want)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            i = first
            while i <= last:
                ok, bgr = cap.read()
                if not ok:
                    break
                if i in want:
                    H, W = bgr.shape[:2]
                    b = bgr.copy()
                    b[955:1045, 480:1920] = 48            # the strip region as the identifier sees it (masked)
                    rgb = Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
                    for r in want[i]:
                        r["crop"] = f"{day}_{race}_f{i}_{r['rider']}.png"
                        canvas(rgb, r["box"], W, H).save(out / "crops" / r["crop"])
                        rows.append(r)
                        kept += 1
                i += 1
            cap.release()
        print(day, "labels found", sum(len(v) for v in found.values()), "kept", kept, flush=True)
    by = Counter(r["class"] for r in rows)
    wrong = Counter(r["class"] for r in rows if r["read_as"] != r["class"] and r["read_p"] >= 0.7)
    lead = Counter(r["class"] for r in rows if r["where"] == "leading")
    print("labels", len(rows), "by class", sorted(by.items()))
    print("the classifier was sure of another number", sorted(wrong.items()))
    print("leading riders", sorted(lead.items()))
    (out / "mined.json").write_text(json.dumps({"days": days, "by_class": dict(sorted(by.items())), "sure_of_another_number": dict(sorted(wrong.items())),
                                                "leading": dict(sorted(lead.items())), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
