"""v12 stage 1 data: riders in every picture of the whole race (clock start to the winner's finish), 2 pictures a second.
usage (server): v12_dense.py DAY [RACE ...]

Same detection and cap classifier as v11 (fast configuration: D0, then D3 when fewer than four riders; no Qwen; cap classifier
cnn/models/caps2_a.pt). The full cap probabilities of every rider are kept, so the top-five and order rules of v12 can be learned and
chosen without running the models again. Pictures are 0.5 s apart, which is the spacing of the program's three pictures (t, t-0.5, t-1.0).
Writes v12/dense/DAY.json: races -> pictures [{to_finish, t, frame, detection, boxes, probs}]. The strip is never read."""
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
from identify_l1 import CNN_WEIGHTS_V11, LeadingGroupIdentifier, video_pts

STEP = 0.5


def main():
    day, only = sys.argv[1], set(sys.argv[2:])
    ident = LeadingGroupIdentifier(fast=True, cap_weights=CNN_WEIGHTS_V11)
    off, tms = fc.official(day), fc.times(day)
    out_path = ROOT / "v12" / "dense" / f"{day}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "v12" / "pts" / day).mkdir(parents=True, exist_ok=True)
    res = {"day": day, "step": STEP, "cap_weights": str(CNN_WEIGHTS_V11.relative_to(ROOT)), "races": {}}
    if out_path.exists():
        res["races"] = json.loads(out_path.read_text())["races"]
    t0 = time.time()
    for race in sorted(off):
        if (only and race not in only) or "error" in off[race] or race in res["races"]:
            continue
        tw = fc.winner_time(off[race])
        vf, how, agree = fc.video_finish(tms.get(race), tw)
        if vf is None:
            continue
        path = ROOT / fc.FOLDER[day] / "videos" / tms[race]["file"]
        pts = video_pts(path, ROOT / "v12" / "pts" / day / f"{race}.json")
        span = vf - tms[race]["S"]
        n = int(span / STEP)
        want = [bisect.bisect_right(pts, vf - k * STEP + 1e-8) - 1 for k in range(n, -1, -1)]
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
        res["races"][race] = {"video_finish": vf, "anchor": how, "clock_agrees": agree, "winner_time": tw, "clock_start": tms[race]["S"], "pictures": pics}
        print(race, len(pics), "pictures, riders per picture", round(float(np.mean([len(p["boxes"]) for p in pics] or [0])), 1), flush=True)
        out_path.write_text(json.dumps(res))
    print("done", day, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
