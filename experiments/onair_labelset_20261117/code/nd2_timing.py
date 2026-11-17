"""Where does the time of the fast program (F2: --fast --fallback --leader-tau 0.5) go? One race, no strip reading. A rough measurement.
usage: nd2_timing.py VIDEO.mp4 PTS.json START END OUT.json
Stages: loading the models, getting the 3 pictures of a time point out of the video file (seek + decode), rider detection, everything else
(cap classifier, the two small models, votes). Also: decoding the same stretch of video front to back without seeking."""
import bisect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "identifier"))
import cv2  # noqa: E402
import identify2  # noqa: E402
import identify_l1 as I  # noqa: E402

video, pts_path, start, end, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
I.wait_gpu()
T = {"load": 0.0, "grab": 0.0, "detect": 0.0, "point_total": 0.0}
t0 = time.perf_counter()
ident = I.LeadingGroupIdentifier("sep06", True, True, 0.5)
T["load"] = time.perf_counter() - t0
orig = identify2.find_riders


def timed(*a, **k):
    s = time.perf_counter()
    r = orig(*a, **k)
    T["detect"] += time.perf_counter() - s
    return r


identify2.find_riders = timed
pts = json.loads(Path(pts_path).read_text())
cap = cv2.VideoCapture(video)


def grab(t):
    i = bisect.bisect_right(pts, t + 1e-8) - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, bgr = cap.read()
    return (bgr if ok else None), i


n, t, per_point = 0, start, []
while t <= end + 1e-6:
    s = time.perf_counter()
    bgr, i = grab(t)
    pics = [bgr] + [grab(pts[i] - off)[0] for off in (0.5, 1.0)]
    g = time.perf_counter() - s
    T["grab"] += g
    s = time.perf_counter()
    ident.point(pts[i], [p for p in pics if p is not None])
    p = time.perf_counter() - s
    T["point_total"] += p
    per_point.append(round(g + p, 3))
    n += 1
    t += 3.0
# the same stretch decoded front to back, no seeking (what a live feed would cost at most)
i0, i1 = bisect.bisect_right(pts, start - 1.0) - 1, bisect.bisect_right(pts, end) - 1
cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
s, frames = time.perf_counter(), 0
for _ in range(i0, i1 + 1):
    ok, _f = cap.read()
    if not ok:
        break
    frames += 1
seq = time.perf_counter() - s
cap.release()
res = {"video": video, "time_points": n, "seconds": {k: round(v, 2) for k, v in T.items()},
       "per_time_point": {"grab_3_pictures": round(T["grab"] / n, 3), "detection": round(T["detect"] / n, 3), "everything_else": round((T["point_total"] - T["detect"]) / n, 3),
                          "total_without_loading": round((T["grab"] + T["point_total"]) / n, 3), "first_point": per_point[0], "median_point": sorted(per_point)[len(per_point) // 2]},
       "sequential_decode": {"frames": frames, "seconds": round(seq, 2), "video_seconds": round(pts[i1] - pts[i0], 2), "frames_per_second": round(frames / seq, 1)}}
Path(out).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
