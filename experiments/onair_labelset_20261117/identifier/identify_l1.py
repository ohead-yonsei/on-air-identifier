"""ON-AIR leading-group identifier as one program. Default since 2026-11-19: v11 = the v9 flow (code name F2) with the cap classifier
retrained on mined labels (caps2/RESULT_KO.md). --v9 gives the default of 2026-11-18 (the earlier cap classifier), --v8 the one before (L1).

From a race broadcast it says, every few seconds, which horse numbers are in the leading group (the front three), using only the picture:
  1. riders: Faster R-CNN, person on a horse; if fewer than 4 are found in a picture, that picture is detected again with the
     small-rider settings (lower score, smaller height, zoom)
  2. numbers: a cap classifier (ResNet-18 trained on 598 hand-labelled caps) scores numbers 1-11, one-to-one assignment within the
     picture; a number is "sure" when the classifier gives it at least 0.7. v8 only: Qwen 9B also names the cap colour of the same
     cap, and the number of that colour gets +1.0 on the classifier's scores before the assignment
  3. front of the field: a small model, trained with the broadcast's running-order strip as the answer, gives each sure rider a
     probability of being in the leading three from the position and size of its box; up to 2 riders with at least 0.6 are taken
  4. a number is stated when it is taken in at least 2 of the 3 most recent pictures (now, 0.5 s ago, 1.0 s ago)
  5. numbers stated at both 3 s and 6 s earlier are carried over
  6. order (added 2026-11-18, see leader/RESULT_KO.md): a second small model, trained with the strip's FIRST number as the answer,
     scores each sure rider. "leader": the top-scoring sure rider of a picture if its score is at least 0.6, when the same number is
     picked in 2 of the 3 pictures (or was the leader at both 3 s and 6 s earlier). "order": for two stated numbers, the one whose
     mean score over the 3 pictures is higher by at least 0.2 is said to be ahead. Neither changes the statements.
  7. the three parts of v9 (night/RESULT_KO.md), all on by default:
     fast          Qwen is never called for the riders; numbers come from the cap classifier alone (code name F1)
     fallback      where steps 4-5 state nothing, one number may be added: the number stated 3 s earlier if it is sure in a current
                   picture; otherwise the sure rider with the highest leading-three probability, if that is at least 0.8
     leader 0.5    threshold for "leader" (step 6 describes v8's 0.6)
     Together they are v9 (code name F2), confirmed on two unused days on 2026-11-18 (night/confirm2/RESULT_KO.md): on those days
     v9 states more (B 94.1 % against 90.1 %) and is a little less precise than v8 (A 94.5 % against 96.4 %).
     --v8          the default before 2026-11-18 (code name L1): Qwen for the riders, no fallback, leader threshold 0.6.
                   With --v8, --fast and --fallback switch single parts on as they did before (--v8 --fast = F1).
     --no-fallback, --leader-tau X   change one part of the default
     The class LeadingGroupIdentifier keeps its old argument defaults (v8); the experiment scripts pass their configuration explicitly.
     Run scripts written before this change (night/run_chain*.sh, live_commentary/*/run_identify.sh) assumed the v8 default.
  8. --v10 (added 2026-11-19, --video only; finish2/RESULT_KO.md): v9 plus two parts for the end of the race, where the strip-trained
     leader model was never taught. Needs the race clock on screen: its last change, minus 0.2 s, is taken as the finish.
     leader      at time points within 36 s of the finish the leader comes from a model learned on official passing times
                 (finish2/models/fs_leader_all.joblib), with the same rule: picked in 2 of the 3 pictures, or carried when the two
                 time points before (within that stretch) agree. Statements and order are unchanged.
     winner      from the 31 pictures of the 6 s before the finish (5 a second): the number the strip-trained leader model picks most
                 often, if in 5 pictures or more; otherwise no call. Written as "winner_call" in the output.
     Nothing here reads the strip, the official result or the sound; the official times were used only to train the model.
     On new days the winner call did not hold (24 calls, 18 right; finish3/RESULT_KO.md): do not rely on it.
  9. --leader-sure X (added 2026-11-19, finish3/RESULT_KO.md): for the leader statement only, a rider counts when its cap probability is
     X or more (default 0.7, the bar used everywhere else). 0.4 was confirmed on two unused days with v10: the leader is named right
     at more time points (44.6 % -> 48.6 %) and wrong at slightly more (5.6 % -> 6.6 %). Statements and order are unchanged.
 10. --cap-weights FILE (added 2026-11-19, caps2/RESULT_KO.md): another cap classifier. cnn/models/caps2_a.pt was trained on the 598 hand
     labels plus 545 labels mined from official order and the strip (riders the current classifier did not read). Confirmed on two unused
     days: the true leader's number is sure more often (62.9 % -> 74.1 % of time points) and the leader is named right more often
     (36.8 % -> 45.8 %; wrong 8.7 % -> 9.4 %); numbers A 93.9 % -> 93.4 %, B 89.5 % -> 91.5 %.
     Since 2026-11-19 it is the default on the command line (v11): on the benchmark test days that gave no mined labels, sentences with a
     wrong number 10.8 % -> 8.1 % and items naming the right leader 51.0 % -> 59.1 % (caps2/DESIGN_BENCH_KO.md, bench_compare.json).
     --v9 and --v8 use the earlier classifier (cnn/models/all_two_days.pt). The class LeadingGroupIdentifier keeps the earlier
     classifier as its argument default, so the experiment scripts run as they did.

The statements never depend on the running-order strip. --strip-check (off by default since 2026-11-19) only reads the strip at every
time point with the local Qwen and writes into the output whether it is on screen and what it shows, for scoring and for marking the
stretch where the first tests were made (order2/RESULT_KO.md, order2b/RESULT_KO.md). Without it no language model is called at all
(unless --v8). The stretches without the strip were measured with official passing times (finish2/RESULT_KO.md).

usage (on the server):
  python identify_l1.py --video RACE.mp4 --out OUT.json [--start S --end E] [--step 3] [--strip-check] [--v8] [--no-fallback] [--leader-tau 0.5] [--v10]
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
TAU_LEADER, MARGIN_ORDER = 0.6, 0.2      # TAU_LEADER is v8's leader threshold
TAU_LEADER_V9 = 0.5                      # v9's leader threshold, the command-line default since 2026-11-18
FALLBACK_SINGLE = 0.8      # --fallback: lowest leading-three probability at which a single rider is stated where nothing else is
TIMER, STRIP = (70, 930, 360, 1055), (480, 955, 1920, 1045)   # at 1920x1080
CNN_WEIGHTS_V11 = ROOT / "cnn" / "models" / "caps2_a.pt"       # the command-line default since 2026-11-19 (v11)
CNN_WEIGHTS, ORDER_MODEL, COLOUR_MODEL = ROOT / "cnn" / "models" / "all_two_days.pt", ROOT / "order2" / "l1_model.joblib", HERE / "model.json"
LEADER_MODEL = ROOT / "leader" / "leader_model.joblib"
FINISH_LEADER_MODEL = ROOT / "finish2" / "models" / "fs_leader_all.joblib"   # --v10: leader in the last stretch, learned on official times
LATE_SECONDS, WINNER_SPAN, WINNER_NEED = 36.0, 6.0, 5                        # --v10: the stretch; winner call from the last 6 s, 5 pictures or more
FAST_ORDER_MODEL, FAST_LEADER_MODEL = ROOT / "night" / "l1_fast.joblib", ROOT / "night" / "leader_fast.joblib"   # --fast: no Qwen (night/DESIGN_KO.md)


class LeadingGroupIdentifier:
    def __init__(self, masks="sep06", fast=False, fallback=False, tau_leader=TAU_LEADER, finish_leader=False, leader_sure=None, cap_weights=None):
        self.cap_weights = Path(cap_weights) if cap_weights else CNN_WEIGHTS
        self.leader_sure = SURE if leader_sure is None else leader_sure      # bar for a rider's number, for the leader statement only
        assert fast or not finish_leader, "the finish-stretch leader model uses the features of the fast configuration"
        self.leader_fs = joblib.load(FINISH_LEADER_MODEL) if finish_leader else None
        self.history_leader_fs = {}   # time -> leader picked by the finish-stretch model (late time points only)
        self.fallback, self.tau_leader = fallback, tau_leader
        self.fast = fast       # fast = configuration F1: Qwen is never called; models trained without the "Qwen agrees" feature
        self.masks = MASKS[masks]
        self.det = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().cuda()
        self.cm = ColourModel(str(COLOUR_MODEL))
        net = resnet18()
        net.fc = nn.Linear(net.fc.in_features, 11)
        net.load_state_dict(torch.load(self.cap_weights, map_location="cpu"))
        self.cnn = net.eval().cuda()
        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.order = joblib.load(FAST_ORDER_MODEL if fast else ORDER_MODEL)
        lm = FAST_LEADER_MODEL if fast else LEADER_MODEL
        self.leader = joblib.load(lm) if lm.exists() else None
        self.history = {}      # time -> set of numbers stated from that time's own pictures
        self.history_leader = {}   # time -> leader picked from that time's own pictures (or None)
        self.qwen_failures = 0

    def picture(self, bgr):
        """One picture -> riders with number, cap probability and probability of being in the leading three."""
        qlog = []
        if self.fast:
            b = bgr.copy()
            for x1, y1, x2, y2 in self.masks:
                b[int(y1 * b.shape[0]):int(y2 * b.shape[0]), int(x1 * b.shape[1]):int(x2 * b.shape[1])] = 48
            found, detection = identify2.find_riders(self.det, b, self.masks, "D0"), "D0"
            if len(found) < MIN_RIDERS:
                found, detection = identify2.find_riders(self.det, b, self.masks, "D3"), "D3"
            rs = [{"box": [round(v, 1) for v in box], "qwen_number": 0} for box, _ in found]
        else:
            rs = identify2.identify(bgr.copy(), self.det, self.cm, self.masks, "D0", qlog)
            detection = "D0"
            if len(rs) < MIN_RIDERS:
                rs = identify2.identify(bgr.copy(), self.det, self.cm, self.masks, "D3", qlog)
                detection = "D3"
        self.qwen_failures += sum(q["error"] is not None for q in qlog)
        if not rs:
            return [], detection, {}, {}
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
            if self.fast:
                f = f[:-1]
            out.append({"number": a[i] + 1, "p_cap": round(p, 4), "sure": p >= SURE, "qwen_number": int(q[i]), "box": rs[i]["box"], "features": f, "_p": p})
        sure = [r for r in out if r["sure"]]
        lead3 = {}             # number -> unrounded leading-three probability, sure riders only (for --fallback)
        if sure:
            pl = self.order.predict_proba(np.array([r["features"] for r in sure]))[:, 1]
            for r, v in zip(sure, pl):
                r["p_lead"] = round(float(v), 4)
                lead3[r["number"]] = float(v)
            for i in np.argsort(-pl)[:K]:
                if pl[i] >= TAU:
                    sure[i]["taken"] = True
        first = {}             # number -> unrounded leader-model score, sure riders only
        cand = sure if self.leader_sure >= SURE else [r for r in out if r["_p"] >= self.leader_sure]     # riders the leader may be picked from
        if cand and self.leader is not None:
            pf = self.leader.predict_proba(np.array([r["features"] for r in cand]))[:, 1]
            for r, v in zip(cand, pf):
                r["p_first"] = round(float(v), 4)
                if r["sure"]:
                    first[r["number"]] = float(v)
            i = int(np.argmax(pf))
            if pf[i] >= self.tau_leader:
                cand[i]["leader_pick"] = True
        if cand and self.leader_fs is not None:
            pf = self.leader_fs.predict_proba(np.array([r["features"] for r in cand]))[:, 1]
            for r, v in zip(cand, pf):
                r["p_first_finish"] = round(float(v), 4)
            i = int(np.argmax(pf))
            if pf[i] >= self.tau_leader:
                cand[i]["leader_pick_finish"] = True
        for r in out:
            del r["features"], r["_p"]
        return out, detection, first, lead3

    def point(self, t, pictures, key=None, late=False):
        """One time point: the pictures at t, t-0.5 s and t-1.0 s (in that order) -> statements, with numbers carried from t-3 s and t-6 s.
        `key` separates races when several are processed by one object. late=True (v10): the time point is in the last stretch, and the
        leader is taken from the finish-stretch model instead of the strip-trained one."""
        votes, best, per_picture, riders_now = Counter(), {}, [], None
        lead_votes, first_scores, top = Counter(), {}, {}
        lead_votes_fs = Counter()
        for bgr in pictures:
            riders, detection, first, lead3 = self.picture(bgr)
            for num, v in lead3.items():
                top[num] = max(top.get(num, 0.0), v)
            for num, v in first.items():
                first_scores.setdefault(num, []).append(v)
            for r in riders:
                if r.get("leader_pick"):
                    lead_votes[r["number"]] += 1
                if r.get("leader_pick_finish"):
                    lead_votes_fs[r["number"]] += 1
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
        added = set()
        if self.fallback and not (now | carried):
            added = prev[0] & set(top)                       # stated 3 s earlier and sure in a current picture
            if not added and top:
                n = max(top, key=top.get)
                if top[n] >= FALLBACK_SINGLE:
                    added = {n}
        detail = [{"number": n, "source": "now", "votes": votes[n], **best[n]} for n in sorted(now)] + [{"number": n, "source": "carried"} for n in sorted(carried)] \
            + [{"number": n, "source": "fallback", "p_lead": round(top[n], 4)} for n in sorted(added)]
        res = {"t": round(float(t), 3), "statements": sorted(now | carried | added), "detail": detail, "pictures": per_picture, "riders_now": riders_now or []}
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
            if late and self.leader_fs is not None:
                now_fs = next((n for n, v in lead_votes_fs.most_common() if v >= 2), None)
                self.history_leader_fs[(key, round(float(t), 3))] = now_fs
                lead_fs, source_fs = now_fs, ("now" if now_fs is not None else None)
                if lead_fs is None:
                    a, b = (self._near(key, t - back, self.history_leader_fs, None) for back in (3.0, 6.0))
                    if a is not None and a == b:
                        lead_fs, source_fs = a, "carried"
                res.update(leader=lead_fs, leader_source=source_fs, leader_model="finish stretch (official times)", leader_strip_model=lead)
        return res

    def winner_call(self, pictures):
        """v10: the pictures of the last seconds before the finish -> (number or None, how many pictures named each number).
        Each picture names the strip-trained model's leader pick or nothing; the number named most often is called if in WINNER_NEED or more."""
        count = Counter()
        for bgr in pictures:
            for r in self.picture(bgr)[0]:
                if r.get("leader_pick"):
                    count[r["number"]] += 1
        top = count.most_common(1)
        return (top[0][0] if top and top[0][1] >= WINNER_NEED else None), dict(count)

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


def race_clock(path):
    """First and last change of the race clock's digits, in video seconds (as code/fn2_times.py and code/ls2_run.py): (S, F)."""
    from fn2_times import series
    fps, on, ch = series(path)
    idx = np.where(on)[0]
    if not len(idx):
        return None, None
    runs, cur = [], [idx[0], idx[0]]
    for i in idx[1:]:
        if i - cur[1] > 3 * fps:
            runs.append(cur)
            cur = [i, i]
        cur[1] = i
    runs.append(cur)
    a, b = max(runs, key=lambda r: r[1] - r[0])
    moving = np.where((ch[a:b + 1] > 0.01) & on[a:b + 1])[0] + a
    tick = [i for k, i in enumerate(moving) if (k > 0 and i - moving[k - 1] <= 0.35 * fps) or (k + 1 < len(moving) and moving[k + 1] - i <= 0.35 * fps)]
    if not tick:
        return None, None
    best, s0 = (tick[0], tick[0]), tick[0]
    for k in range(1, len(tick)):
        if tick[k] - tick[k - 1] > 1.0 * fps:
            s0 = tick[k]
        if tick[k] - s0 > best[1] - best[0]:
            best = (s0, tick[k])
    return round(float(best[0] / fps), 3), round(float(best[1] / fps), 3)


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
    if a.strip_check:
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

    finish = None
    if a.v10:
        S, F = race_clock(path)
        assert F is not None, "--v10 needs the race clock on screen"
        finish = round(F - 0.2, 3)        # the last change of the digits comes about 0.2 s after the clock stops
        print(f"race clock: first change {S}, last change {F}; finish taken as {finish}", flush=True)
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
        if a.strip_check:
            x1, y1, x2, y2 = STRIP
            strip, _ = read_strip(bgr[y1:y2, x1:x2])
        pics = [bgr] + [grab(pts[i] - off)[0] for off in (0.5, 1.0)]
        late = finish is not None and -1.0 <= finish - t <= LATE_SECONDS
        res = ident.point(pts[i], [p for p in pics if p is not None], late=late)
        res["asked_t"] = t
        if finish is not None:
            res["to_finish"], res["late"] = round(finish - t, 2), late
        if a.strip_check:
            res["strip_on_screen"] = strip is not None
            res["strip"] = strip
            res["validated_segment"] = strip is not None
        points.append(res)
        print(f"t {res['t']:7.2f}  statements {res['statements']}" + ("" if not a.strip_check else f"  strip {'on screen ' + str(strip) if strip else 'absent (statements not validated here)'}"), flush=True)
        t += a.step
    extra = {}
    if finish is not None:
        number, count = ident.winner_call([b for b in (grab(finish - WINNER_SPAN + 0.2 * k)[0] for k in range(int(WINNER_SPAN * 5) + 1)) if b is not None])
        extra = {"finish": {"clock_first_change": S, "clock_last_change": F, "finish": finish, "late_seconds": LATE_SECONDS},
                 "winner_call": {"number": number, "pictures_naming_each": count, "span": WINNER_SPAN, "need": WINNER_NEED}}
        print("winner call:", number, count, flush=True)
    cap.release()
    return {"video": str(path), "live_timer_segment": live, "start": start, "end": end, "step": a.step, "points": points, **extra}


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
    ap.add_argument("--strip-check", action="store_true", help="also read the running-order strip at every time point (local Qwen) and write it into the output; the statements do not use it")
    ap.add_argument("--no-strip-check", action="store_true", help="accepted for old command lines; the strip check is off unless --strip-check is given")
    ap.add_argument("--masks", default="sep06", choices=list(MASKS))
    ap.add_argument("--v9", action="store_true", help="the default of 2026-11-18: the same flow with the earlier cap classifier")
    ap.add_argument("--v8", action="store_true", help="the default before 2026-11-18 (L1): Qwen for the riders, no fallback, leader threshold 0.6, the earlier cap classifier")
    ap.add_argument("--fast", action="store_true", help="no Qwen for the riders. On by default; only changes something together with --v8")
    ap.add_argument("--fallback", action="store_true", help="where nothing is stated, add one number. On by default; only changes something together with --v8")
    ap.add_argument("--no-fallback", action="store_true", help="never add a number where nothing is stated")
    ap.add_argument("--leader-sure", type=float, default=None, help="for the leader statement only: lowest cap probability at which a rider counts (default 0.7; 0.4 confirmed on unused days)")
    ap.add_argument("--cap-weights", default=None, help="another cap classifier (default cnn/models/all_two_days.pt; cnn/models/caps2_a.pt is the one retrained with mined labels)")
    ap.add_argument("--v10", action="store_true", help="v9 + leader in the last 36 s from the model learned on official times + winner call (--video only)")
    ap.add_argument("--leader-tau", type=float, default=None, help="threshold for the leader (default 0.5; 0.6 with --v8)")
    a = ap.parse_args()
    assert not (a.v8 and a.v9), "give at most one of --v8 and --v9"
    if a.cap_weights is None:
        a.cap_weights = str(CNN_WEIGHTS if (a.v8 or a.v9) else CNN_WEIGHTS_V11)
    a.fast = a.fast or not a.v8
    a.fallback = (a.fallback or not a.v8) and not a.no_fallback
    if a.leader_tau is None:
        a.leader_tau = TAU_LEADER if a.v8 else TAU_LEADER_V9
    assert not a.v10 or (a.video and a.fast and a.fallback and a.leader_tau == TAU_LEADER_V9), "--v10 is v9 plus the finish parts: use it with --video and without --v8 / --no-fallback / --leader-tau"
    assert bool(a.video) != bool(a.day_dir), "give either --video or --day-dir"
    wait_gpu()
    started = time.perf_counter()
    assert a.leader_sure is None or a.fast, "--leader-sure was tested with the fast configuration only (not with --v8)"
    ident = LeadingGroupIdentifier(a.masks, a.fast, a.fallback, a.leader_tau, finish_leader=a.v10, leader_sure=a.leader_sure, cap_weights=a.cap_weights)
    name = "F2" if (a.fast and a.fallback and a.leader_tau == 0.5) else ("F1" if a.fast else "L1") + ("+fallback" if a.fallback else "") + ("" if a.leader_tau == TAU_LEADER else f"+leader{a.leader_tau}")
    res = run_video(a, ident) if a.video else run_day_dir(a, ident)
    if Path(a.cap_weights).resolve() == CNN_WEIGHTS_V11.resolve():
        name += "+caps2"
    elif Path(a.cap_weights).resolve() != CNN_WEIGHTS.resolve():
        name += "+" + Path(a.cap_weights).stem
    if a.v10:
        name += "+finish"
    res.update(configuration=name, version={"F2+caps2": "v11", "F2+caps2+finish": "v11 with the finish parts", "F2": "v9", "L1": "v8", "F2+finish": "v10"}.get(name), settings={"sure": SURE, "tau": TAU, "k": K, "min_riders_for_usual_detection": MIN_RIDERS,
                                              "order_rules": ident.leader is not None, "tau_leader": a.leader_tau, "margin_order": MARGIN_ORDER,
                                              "fast": a.fast, "fallback": a.fallback, "fallback_single": FALLBACK_SINGLE, "leader_sure": ident.leader_sure, "cap_weights": ident.cap_weights.name},
               qwen_failures=ident.qwen_failures, seconds=round(time.perf_counter() - started, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("time points", len(res["points"]), "qwen failures", ident.qwen_failures, "seconds", res["seconds"])
    if a.fast and not a.strip_check:
        return              # no language model was called
    try:
        urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/generate", data=json.dumps({"model": MODEL, "keep_alive": 0}).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=60).read()
    except Exception as e:
        print("could not unload the model:", repr(e))


if __name__ == "__main__":
    main()
