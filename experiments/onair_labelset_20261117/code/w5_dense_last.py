"""winner: riders in the pictures of the last seconds of each race, 5 pictures a second, with the v11 cap classifier.
usage: w5_dense_last.py DAY [RACE ...]

As fn6_dense.py (same detection, same cap picture), but the classifier is the program default of v11 (cnn/models/caps2_a.pt) and only
the last 8 s are kept. Writes winner/dense/DAY.json: races -> pictures [{to_finish, t, frame, detection, boxes, probs}].
The finish is placed as in every measurement here (fn_common.video_finish: clock start + official winning time). The strip is never read."""
import bisect
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "identifier"))
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
from fn6_dense import riders
from identify_l1 import LeadingGroupIdentifier, video_pts

SPAN, STEP = 8.0, 0.2
WEIGHTS = ROOT / "cnn" / "models" / "caps2_a.pt"


def main():
    day, only = sys.argv[1], set(sys.argv[2:])
    fc.FOLDER.setdefault(day, day)
    ident = LeadingGroupIdentifier(fast=True, cap_weights=WEIGHTS)
    off, tms = fc.official(day), fc.times(day)
    out_path = ROOT / "winner" / "dense" / f"{day}.json"
    (ROOT / "winner" / "pts" / day).mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = {"day": day, "span": SPAN, "step": STEP, "cap_weights": WEIGHTS.name, "races": {}}
    t0 = time.time()
    for race in sorted(off):
        if (only and race not in only) or "error" in off[race]:
            continue
        tw = fc.winner_time(off[race])
        vf, how, agree = fc.video_finish(tms.get(race), tw)
        if vf is None:
            print(race, "no finish time:", how, flush=True)
            continue
        path = ROOT / fc.FOLDER[day] / "videos" / tms[race]["file"]
        pts = video_pts(path, ROOT / "winner" / "pts" / day / f"{race}.json")
        n = int(round(SPAN / STEP))
        want = [bisect.bisect_right(pts, vf - SPAN + k * STEP + 1e-8) - 1 for k in range(n + 1)]
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(want[0], 0))
        i, pics, wanted = max(want[0], 0), [], set(want)
        while i <= want[-1]:
            ok, bgr = cap.read()
            if not ok:
                break
            if i in wanted:
                detection, boxes, P = riders(ident, bgr)
                pics.append({"to_finish": round(vf - pts[i], 3), "t": pts[i], "frame": i, "detection": detection, "boxes": boxes, "probs": P})
            i += 1
        cap.release()
        res["races"][race] = {"video_finish": vf, "anchor": how, "clock_agrees": agree, "winner_time": tw, "pictures": pics}
        print(race, len(pics), "pictures, riders per picture", round(float(np.mean([len(p["boxes"]) for p in pics])), 1), flush=True)
        out_path.write_text(json.dumps(res))
    print("done", day, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
