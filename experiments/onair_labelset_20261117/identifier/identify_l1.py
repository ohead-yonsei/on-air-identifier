"""ON-AIR leading-group identifier, configuration L1, as one program.

From a race broadcast it says, every few seconds, which horse numbers are in the leading group (the front three), using only the picture:
  1. riders: Faster R-CNN, person on a horse; if fewer than 4 are found in a picture, that picture is detected again with the
     small-rider settings (lower score, smaller height, zoom)
  2. numbers: Qwen 9B names the cap colour, a cap classifier (ResNet-18 trained on 598 hand-labelled caps) scores numbers 1-11,
     one-to-one assignment within the picture; a number is "sure" when the classifier gives it at least 0.7
  3. front of the field: a small model, trained with the broadcast's running-order strip as the answer, gives each sure rider a
     probability of being in the leading three from the position and size of its box; up to 2 riders with at least 0.6 are taken
  4. a number is stated when it is taken in at least 2 of the 3 most recent pictures (now, 0.5 s ago, 1.0 s ago)
  5. numbers stated at both 3 s and 6 s earlier are carried over
  6. order (added 2026-11-18, see leader/RESULT_KO.md): a second small model, trained with the strip's FIRST number as the answer,
     scores each sure rider. "leader": the top-scoring sure rider of a picture if its score is at least 0.6, when the same number is
     picked in 2 of the 3 pictures (or was the leader at both 3 s and 6 s earlier). "order": for two stated numbers, the one whose
     mean score over the 3 pictures is higher by at least 0.2 is said to be ahead. Neither changes the statements.

Tested only while the running-order strip is on screen (see order2/RESULT_KO.md, order2b/RESULT_KO.md). Where the strip is absent
(about the last 30 s, and just after the start) the statements are not reliable; the output marks those time points.

usage (on the server):
  python identify_l1.py --video RACE.mp4 --out OUT.json [--start S --end E] [--step 3] [--no-strip-check]
  python identify_l1.py --day-dir otherday7 --out OUT.json [--races R01 R02]      # frames already extracted (manifest.json, frames/, frames_past/)
"""
import argparse
import bisect
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / "model_cache"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code"))

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

import identify2
from identify import ENDPOINT, MASKS, MODEL, ColourModel, wait_gpu
from s3_classify import region as cap_region

SURE, TAU, K, MIN_RIDERS = 0.7, 0.6, 2, 4
TAU_LEADER, MARGIN_ORDER = 0.6, 0.2
TIMER, STRIP = (70, 930, 360, 1055), (480, 955, 1920, 1045)   # at 1920x1080
CNN_WEIGHTS, ORDER_MODEL, COLOUR_MODEL = ROOT / "cnn" / "models" / "all_two_days.pt", ROOT / "order2" / "l1_model.joblib", HERE / "model.json"
LEADER_MODEL = ROOT / "leader" / "leader_model.joblib"


class LeadingGroupIdentifier:
    def __init__(self, masks="sep06"):
        self.masks = MASKS[masks]
        self.det = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().cuda()
        self.cm = ColourModel(str(COLOUR_MODEL))
        net = resnet18()
        net.fc = nn.Linear(net.fc.in_features, 11)
        net.load_state_dict(torch.load(CNN_WEIGHTS, map_location="cpu"))
        self.cnn = net.eval().cuda()
        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.order = joblib.load(ORDER_MODEL)
        self.leader = joblib.load(LEADER_MODEL) if LEADER_MODEL.exists() else None
        self.history = {}      # time -> set of numbers stated from that time's own pictures
        self.history_leader = {}   # time -> leader picked from that time's own pictures (or None)
        self.qwen_failures = 0

    def picture(self, bgr):
        """One picture -> riders with number, cap probability and probability of being in the leading three."""
        qlog = []
        rs = identify2.identify(bgr.copy(), self.det, self.cm, self.masks, "D0", qlog)
        detection = "D0"
        if len(rs) < MIN_RIDERS:
            rs = identify2.identify(bgr.copy(), self.det, self.cm, self.masks, "D3", qlog)
            detection = "D3"
        self.qwen_failures += sum(q["error"] is not None for q in qlog)
        if not rs:
            return [], detection, {}
        H, W = bgr.shape[:2]
        masked = bgr.copy()
        for x1, y1, x2, y2 in self.masks:
            masked[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = 48
        rgb = Image.fromarray(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        xs = []
        for r in rs:
            can = Image.new("RGB", (224, 224), (96, 96, 96))
            patch = ImageOps.contain(rgb.crop(cap_region(r["box"], "R0_35", W, H)), (224, 224), Image.Resampling.LANCZOS)
            can.paste(patch, ((224 - patch.width) // 2, (224 - patch.height) // 2))
            xs.append(self.tf(can))
        with torch.inference_mode():
            P = np.array(torch.softmax(self.cnn(torch.stack(xs).cuda()), 1).cpu().numpy().round(5).tolist())
        Sc = P.copy()
        q = [r["qwen_number"] for r in rs]
        for i, n in enumerate(q):
            if n:
                Sc[i, n - 1] += 1.0
        ri, cj = linear_sum_assignment(-Sc)
        a = {int(i): int(j) for i, j in zip(ri, cj)}
        boxes = np.array([r["box"] for r in rs], float)
        cx, yb, h = (boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3], boxes[:, 3] - boxes[:, 1]
        n = len(rs)
        ry, rxr, rxl, rh = (np.argsort(np.argsort(v)) for v in (-yb, -cx, cx, -h))
        corr = float(np.corrcoef(cx, yb)[0, 1]) if n >= 3 and cx.std() > 1 and yb.std() > 1 else 0.0
        out = []
        for i in range(n):
            if i not in a:
                continue
            p = float(P[i, a[i]])
            f = [ry[i], rxr[i], rxl[i], rh[i], ry[i] / n, rxr[i] / n, rh[i] / n, n, yb[i] / 1080, cx[i] / 1920, h[i] / 1080, h[i] / h.max(),
                 (yb[i] - yb.mean()) / 1080, (cx[i] - cx.mean()) / 1920, cx.std() / 1920, yb.std() / 1080, corr, h.mean() / 1080, p, float(q[i] == a[i] + 1)]
            out.append({"number": a[i] + 1, "p_cap": round(p, 4), "sure": p >= SURE, "qwen_number": int(q[i]), "box": rs[i]["box"], "features": f})
        sure = [r for r in out if r["sure"]]
        if sure:
            pl = self.order.predict_proba(np.array([r["features"] for r in sure]))[:, 1]
            for r, v in zip(sure, pl):
                r["p_lead"] = round(float(v), 4)
            for i in np.argsort(-pl)[:K]:
                if pl[i] >= TAU:
                    sure[i]["taken"] = True
        first = {}             # number -> unrounded leader-model score, sure riders only
        if sure and self.leader is not None:
            pf = self.leader.predict_proba(np.array([r["features"] for r in sure]))[:, 1]
            for r, v in zip(sure, pf):
                r["p_first"] = round(float(v), 4)
                first[r["number"]] = float(v)
            i = int(np.argmax(pf))
            if pf[i] >= TAU_LEADER:
                sure[i]["leader_pick"] = True
        for r in out:
            del r["features"]
        return out, detection, first

    def point(self, t, pictures, key=None):
        """One time point: the pictures at t, t-0.5 s and t-1.0 s (in that order) -> statements, with numbers carried from t-3 s and t-6 s.
        `key` separates races when several are processed by one object."""
        votes, best, per_picture, riders_now = Counter(), {}, [], None
        lead_votes, first_scores = Counter(), {}
        for bgr in pictures:
            riders, detection, first = self.picture(bgr)
            for num, v in first.items():
                first_scores.setdefault(num, []).append(v)
            for r in riders:
                if r.get("leader_pick"):
                    lead_votes[r["number"]] += 1
            if riders_now is None:
                riders_now = riders      # the riders of the newest picture, kept for drawing and inspection
            per_picture.append({"riders": len(riders), "detection": detection, "sure": sum(r["sure"] for r in riders)})
            for num in {r["number"] for r in riders if r.get("taken")}:
                votes[num] += 1
            for r in riders:
                if r.get("taken") and (r["number"] not in best or r["p_lead"] > best[r["number"]]["p_lead"]):
                    best[r["number"]] = {"p_lead": r["p_lead"], "p_cap": r["p_cap"]}
        now = {n for n, v in votes.items() if v >= 2}
        self.history[(key, round(float(t), 3))] = now
        prev = [self._near(key, t - back) for back in (3.0, 6.0)]
        carried = (prev[0] & prev[1]) - now
        detail = [{"number": n, "source": "now", "votes": votes[n], **best[n]} for n in sorted(now)] + [{"number": n, "source": "carried"} for n in sorted(carried)]
        res = {"t": round(float(t), 3), "statements": sorted(now | carried), "detail": detail, "pictures": per_picture, "riders_now": riders_now or []}
        if self.leader is not None:
            picked = [n for n, v in lead_votes.items() if v >= 2]
            lead = picked[0] if picked else None
            self.history_leader[(key, round(float(t), 3))] = lead
            source = "now" if lead is not None else None
            if lead is None:
                a, b = (self._near(key, t - back, self.history_leader, None) for back in (3.0, 6.0))
                if a is not None and a == b:
                    lead, source = a, "carried"
            sc = {n: sum(v) / len(v) for n, v in first_scores.items()}
            nums, order = sorted(n for n in res["statements"] if n in sc), []
            for i, a in enumerate(nums):
                for b in nums[i + 1:]:
                    if abs(sc[a] - sc[b]) >= MARGIN_ORDER:
                        hi, lo = (a, b) if sc[a] > sc[b] else (b, a)
                        order.append({"ahead": hi, "behind": lo, "gap": round(sc[hi] - sc[lo], 4)})
            res.update(leader=lead, leader_source=source, order=order)
        return res

    def _near(self, key, t, store=None, default=None):
        store = self.history if store is None else store
        cands = [(abs(tt - t), s) for (k, tt), s in store.items() if k == key and abs(tt - t) <= 0.6]
        return min(cands, key=lambda c: c[0])[1] if cands else (set() if store is self.history else default)


def video_pts(path, cache):
    if cache.exists():
        return json.loads(cache.read_text())
    raw = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)])
    pts = [float(x["best_effort_timestamp_time"]) for x in json.loads(raw)["frames"]]
    cache.write_text(json.dumps(pts))
    return pts


def live_segment(grab, last_second):
    """Seconds during which the race timer is on screen (the longest run)."""
    on = []
    for t in range(1, last_second):
        bgr, _ = grab(t)
        if bgr is None:
            break
        x1, y1, x2, y2 = TIMER
        hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        if float(((h >= 100) & (h <= 125) & (s > 140) & (v > 90)).mean()) >= 0.06 and float(((s < 40) & (v > 215)).mean()) >= 0.15:
            on.append(t)
    runs, cur = [], []
    for t in on:
        if cur and t - cur[-1] > 3:
            runs.append(cur)
            cur = []
        cur.append(t)
    if cur:
        runs.append(cur)
    best = max(runs, key=len) if runs else []
    return (best[0], best[-1]) if best else (None, None)


def run_video(a, ident):
    from j1_strip_reader import read_strip
    path = Path(a.video)
    out_path = Path(a.out)
    pts = video_pts(path, out_path.with_suffix(".pts.json"))
    cap = cv2.VideoCapture(str(path))
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (1920, 1080), "the graphics positions assume a 1920x1080 broadcast"

    def grab(t):
        i = bisect.bisect_right(pts, t + 1e-8) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        return (bgr if ok else None), i

    start, end = a.start, a.end
    live = (None, None)
    if start is None or end is None:
        live = live_segment(grab, int(pts[-1]))
        assert live[0] is not None, "no race timer found; give --start and --end"
        start = live[0] + 22 if start is None else start     # the first 22 s after the timer appears were never tested
        end = live[1] if end is None else end
    points, t = [], float(start)
    while t <= end + 1e-6:
        bgr, i = grab(t)
        if bgr is None:
            break
        strip = None
        if not a.no_strip_check:
            x1, y1, x2, y2 = STRIP
            strip, _ = read_strip(bgr[y1:y2, x1:x2])
        pics = [bgr] + [grab(pts[i] - off)[0] for off in (0.5, 1.0)]
        res = ident.point(pts[i], [p for p in pics if p is not None])
        res["asked_t"] = t
        if not a.no_strip_check:
            res["strip_on_screen"] = strip is not None
            res["strip"] = strip
            res["validated_segment"] = strip is not None
        points.append(res)
        print(f"t {res['t']:7.2f}  statements {res['statements']}" + ("" if a.no_strip_check else f"  strip {'on screen ' + str(strip) if strip else 'absent (statements not validated here)'}"), flush=True)
        t += a.step
    cap.release()
    return {"video": str(path), "live_timer_segment": live, "start": start, "end": end, "step": a.step, "points": points}


def run_day_dir(a, ident):
    O = ROOT / a.day_dir
    man = json.loads((O / "manifest.json").read_text())
    points = []
    for m in man["frames"]:
        if a.races and m["race"] not in a.races:
            continue
        race, t = m["frame"].split("_t")
        pics = [cv2.imread(str(O / "frames" / f"{m['frame']}.png"))] + [cv2.imread(str(O / "frames_past" / f"{m['frame']}_{tag}.png")) for tag in ("m05", "m10")]
        res = ident.point(float(t), pics, key=race)
        res.update(frame=m["frame"], race=race, strip=m.get("strip"))
        points.append(res)
        print(m["frame"], "statements", res["statements"], "strip", m.get("strip"), flush=True)
    return {"day_dir": a.day_dir, "points": points}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--day-dir")
    ap.add_argument("--races", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--step", type=float, default=3.0)
    ap.add_argument("--no-strip-check", action="store_true")
    ap.add_argument("--masks", default="sep06", choices=list(MASKS))
    a = ap.parse_args()
    assert bool(a.video) != bool(a.day_dir), "give either --video or --day-dir"
    wait_gpu()
    started = time.perf_counter()
    ident = LeadingGroupIdentifier(a.masks)
    res = run_video(a, ident) if a.video else run_day_dir(a, ident)
    res.update(configuration="L1", settings={"sure": SURE, "tau": TAU, "k": K, "min_riders_for_usual_detection": MIN_RIDERS,
                                              "order_rules": ident.leader is not None, "tau_leader": TAU_LEADER, "margin_order": MARGIN_ORDER},
               qwen_failures=ident.qwen_failures, seconds=round(time.perf_counter() - started, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("time points", len(res["points"]), "qwen failures", ident.qwen_failures, "seconds", res["seconds"])
    try:
        urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/generate", data=json.dumps({"model": MODEL, "keep_alive": 0}).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=60).read()
    except Exception as e:
        print("could not unload the model:", repr(e))


if __name__ == "__main__":
    main()
