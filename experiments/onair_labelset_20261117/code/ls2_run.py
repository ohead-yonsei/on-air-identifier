"""Script demo, step 2: run the identifier over one race video as a live system could, and save what the commentary may use.
usage: ls2_run.py DEMO_DIR        (DEMO_DIR has videos/*.video.mp4 and racecard.json)

Nothing here reads the running-order strip, the official result or the sound.
  clock      start S and stop F of the race clock on screen (as fn2_times.py: the digits inside the blue pill)
  points     3 s and 6 s after the start: no identifier (picture kept for the page). Then every 3 s from 9 s after the start to the
             clock stop, plus one at the stop: v9 (identify_l1.py --fast --fallback
             --leader-tau 0.5) on the pictures at t, t-0.5 s, t-1.0 s
  leader     from 36 s before the expected finish on (expected from the race distance, as the earlier demo did) the leader comes
             from the model learned on official times (finish2/models/fs_leader_all.joblib): picked in 2 of the 3 pictures, or
             carried when the two time points before agree
  winner     at the clock stop: every picture of the 6 s before it, 5 a second; the number picked as leader most often, if in 5
             pictures or more (finish2: rule P:strip:6.0:5:1.0)
Writes DEMO_DIR/ident.json and DEMO_DIR/frames/tNNN.jpg (640x360, the riders the identifier is sure of outlined)."""
import bisect
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "identifier"))
sys.path.insert(0, str(ROOT / "code"))
import fn_tracks as ft
from fn2_times import series
from fn6_dense import riders
from identify_l1 import LeadingGroupIdentifier, video_pts

D = ROOT / sys.argv[1]
card = json.loads((D / "racecard.json").read_text())
video = next((D / "videos").glob("*.video.mp4"))


def expected_seconds(d):
    return d * (0.062 if d <= 1200 else 0.0635 if d <= 1400 else 0.066)


def clock(path):
    fps, on, ch = series(path)
    idx = np.where(on)[0]
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
    best, s0 = (tick[0], tick[0]), tick[0]
    for k in range(1, len(tick)):
        if tick[k] - tick[k - 1] > 1.0 * fps:
            s0 = tick[k]
        if tick[k] - s0 > best[1] - best[0]:
            best = (s0, tick[k])
    return round(float(best[0] / fps), 3), round(float(best[1] / fps), 3)


S, F = clock(video)
start, stop = S - 0.1, F - 0.2          # the clock shows 0:00.1 at S; its last change comes about 0.2 s after it stops
T = expected_seconds(card["distance"])
print("clock start", start, "stop", stop, "ran", round(stop - start, 1), "s; expected from the distance", round(T, 1), flush=True)

ident = LeadingGroupIdentifier(fast=True, fallback=True, tau_leader=0.5)
order_model = joblib.load(ROOT / "night" / "l1_fast.joblib")
leader_strip = joblib.load(ROOT / "night" / "leader_fast.joblib")
leader_fs = joblib.load(ROOT / "finish2" / "models" / "fs_leader_all.joblib")
pts = video_pts(video, D / "pts.json")
cap = cv2.VideoCapture(str(video))
(D / "frames").mkdir(exist_ok=True)


def grab(t):
    i = bisect.bisect_right(pts, t + 1e-8) - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(i, 0))
    ok, bgr = cap.read()
    return (bgr if ok else None), i


def picks(bgr, leader_model):
    det, boxes, P = riders(ident, bgr)
    pic = {"boxes": boxes, "probs": P}
    who = ft.identify([pic], [list(range(len(boxes)))], memory=False)[0]
    return ft.decide(pic, who, order_model, leader_model)


times = [t for t in (start + 9 + 3 * k for k in range(int((stop - start - 9) // 3) + 1)) if stop - t > 1.5] + [stop]      # the last one is the clock stop
points, hist_fs = [], {}
for e0 in (3, 6):                     # right after the start the identifier is not used (never tested there): picture only, for the page
    bgr, _ = grab(start + e0)
    name = f"t{e0:03d}.jpg"
    cv2.imwrite(str(D / "frames" / name), cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
    points.append({"t": round(float(start + e0), 3), "elapsed": float(e0), "statements": [], "detail": [], "leader": None, "leader_source": None, "leader_model": None,
                   "leader_v9": None, "order": [], "late": False, "at_clock_stop": False, "frame": name, "riders": [], "early": True})
for t in times:
    bgr, i = grab(t)
    pics = [p for p in [bgr] + [grab(pts[i] - o)[0] for o in (0.5, 1.0)] if p is not None]
    r = ident.point(pts[i], pics, key="race")
    elapsed = round(float(t - start), 1)
    late = bool(elapsed >= T - 36)
    lead, source, model = r.get("leader"), r.get("leader_source"), "strip-trained (v9)"
    if late:
        votes = Counter(d["pick"] for d in (picks(p, leader_fs) for p in pics) if d["pick"] is not None)
        now = next((n for n, v in votes.most_common() if v >= 2), None)
        hist_fs[round(t, 1)] = now
        lead, source, model = now, ("now" if now is not None else None), "learned from official times (v10 piece)"
        if lead is None:
            a, b = hist_fs.get(round(t - 3, 1)), hist_fs.get(round(t - 6, 1))
            if a is not None and a == b:
                lead, source = a, "carried"
    img = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
    for rd in r["riders_now"]:
        if rd["sure"]:
            x1, y1, x2, y2 = [int(v / 3) for v in rd["box"]]
            hot = rd["number"] in r["statements"] or rd["number"] == lead
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 220, 60) if hot else (200, 200, 200), 2 if hot else 1)
            cv2.putText(img, str(rd["number"]), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60) if hot else (200, 200, 200), 2 if hot else 1)
    name = f"t{int(round(elapsed)):03d}.jpg"
    cv2.imwrite(str(D / "frames" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    points.append({"t": round(float(t), 3), "elapsed": elapsed, "statements": r["statements"], "detail": r["detail"], "leader": lead, "leader_source": source,
                   "leader_model": model, "leader_v9": r.get("leader"), "order": r.get("order", []), "late": late, "at_clock_stop": bool(abs(t - stop) < 0.05),
                   "frame": name, "riders": [x["riders"] for x in r["pictures"]]})
    print(f"{elapsed:6.1f}s  said {r['statements']}  leader {lead} ({source})  order {[(o['ahead'], o['behind']) for o in r.get('order', [])]}", flush=True)

# winner call from the 6 s before the clock stop
count = Counter()
for k in range(31):
    bgr, _ = grab(stop - 6.0 + 0.2 * k)
    if bgr is not None:
        p = picks(bgr, leader_strip)["pick"]
        if p is not None:
            count[p] += 1
top = count.most_common(2)
call = top[0][0] if top and top[0][1] >= 5 else None
print("winner call:", call, dict(count), flush=True)
cap.release()
(D / "ident.json").write_text(json.dumps({"video": video.name, "clock": {"start": start, "stop": stop, "S": S, "F": F}, "expected_seconds": round(T, 1),
                                         "configuration": "v9 (F2); leader in the last 36 s from the model learned on official times; winner call from dense pictures",
                                         "points": points, "winner_call": {"number": call, "pictures_naming_each": dict(count)}}, ensure_ascii=False, indent=1))
