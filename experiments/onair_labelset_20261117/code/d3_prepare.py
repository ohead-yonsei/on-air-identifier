"""Third-day test (2026-11-07), step 1; same as od1_prepare.py with the folder and date changed.

Other-day test, step 1: verify videos, find live segments, build strip frames and their 0.5 s / 1.0 s predecessors.

Same procedure as for 2026-09-06 (s1_live_segments.py, k1_build_stripset.py, r1_past_frames.py). See otherday2/DESIGN_KO.md.
"""
import bisect
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "otherday2"
sys.path.insert(0, str(ROOT / "code"))
from j1_strip_reader import ENDPOINT, MODEL, read_strip  # noqa: E402

TIMER = (70, 930, 360, 1055)
STRIP = (480, 955, 1920, 1045)
MASKS = [(0, 0, 360, 210), (1710, 0, 1920, 180), (0, 930, 1920, 1080)]
ids = json.loads((O / "video_ids.json").read_text())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


sources, manifest, past = [], [], []
for d in ("frames", "frames_past", "evidence"):
    (O / d).mkdir(exist_ok=True)
for v in ids["videos"]:
    race = v["race"]
    path = O / "videos" / f"{race}_{v['id']}.mp4"
    info = json.loads((O / "videos" / f"{race}_{v['id']}.info.json").read_text())
    ok_title = f"2026.11.07 R{int(race[1:])}" in info.get("title", "") and "(Seoul)" in info.get("title", "")
    pr = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]))
    vs = [s for s in pr["streams"] if s["codec_type"] == "video"][0]
    src = {"race": race, "id": v["id"], "file": str(path.relative_to(O)), "title": info.get("title"), "channel": info.get("channel"), "title_ok": ok_title,
           "width": vs["width"], "height": vs["height"], "fps": vs["avg_frame_rate"], "codec": vs["codec_name"], "duration": float(pr["format"]["duration"]),
           "audio_streams": sum(s["codec_type"] == "audio" for s in pr["streams"]), "size": path.stat().st_size, "sha256": sha(path)}
    sources.append(src)
    if not ok_title or (vs["width"], vs["height"]) != (1920, 1080):
        print(race, "SKIPPED", src["title"], vs["width"], vs["height"], flush=True)
        continue
    pp = O / "evidence" / f"{race}_pts.json"
    if pp.exists():
        pts = json.loads(pp.read_text())
    else:
        raw = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)])
        pts = [float(x["best_effort_timestamp_time"]) for x in json.loads(raw)["frames"]]
        pp.write_text(json.dumps(pts))
    cap = cv2.VideoCapture(str(path))

    def grab(t):
        i = bisect.bisect_right(pts, t + 1e-8) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        return (bgr if ok else None), i

    on = []
    for t in range(1, int(pts[-1])):
        bgr, _ = grab(t)
        if bgr is None:
            break
        x1, y1, x2, y2 = TIMER
        hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        h, s, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        if float(((h >= 100) & (h <= 125) & (s > 140) & (val > 90)).mean()) >= 0.06 and float(((s < 40) & (val > 215)).mean()) >= 0.15:
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
    src["live_start"], src["live_end"] = (best[0], best[-1]) if best else (None, None)
    kept = 0
    if best:
        for t in range(best[0] + 22, best[-1] - 8 + 1, 3):
            bgr, i = grab(t)
            if bgr is None:
                continue
            x1, y1, x2, y2 = STRIP
            nums, _ = read_strip(bgr[y1:y2, x1:x2])
            if nums is None:
                continue
            fid = f"{race}_t{t:03d}"
            for mx1, my1, mx2, my2 in MASKS:
                bgr[my1:my2, mx1:mx2] = 48
            cv2.imwrite(str(O / "frames" / f"{fid}.png"), bgr)
            manifest.append({"frame": fid, "race": race, "pts": pts[i], "decoded_index": i, "strip": nums})
            for tag, off in (("m05", 0.5), ("m10", 1.0)):
                pb, pi = grab(pts[i] - off)
                assert pb is not None and pts[pi] < pts[i]
                for mx1, my1, mx2, my2 in MASKS:
                    pb[my1:my2, mx1:mx2] = 48
                cv2.imwrite(str(O / "frames_past" / f"{fid}_{tag}.png"), pb)
                past.append({"name": f"{fid}_{tag}", "frame": fid, "pts": pts[pi]})
            kept += 1
    cap.release()
    print(race, "live", (src["live_start"], src["live_end"]), "strip frames", kept, flush=True)
(O / "video_sources.json").write_text(json.dumps(sources, indent=1))
(O / "manifest.json").write_text(json.dumps({"day": ids["day"], "frames": manifest, "past": past}, indent=1))
print("videos", len(sources), "strip frames", len(manifest), "past frames", len(past))
try:
    urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/generate", data=json.dumps({"model": MODEL, "keep_alive": 0}).encode(), headers={"Content-Type": "application/json"}), timeout=60).read()
except Exception:
    pass
