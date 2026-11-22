"""live4 exploration: does the race clock appear on screen, and does the live runner's clock reader find the start and the finish?
usage: lv5_clock_scan.py OUT.json        (server; reads every race video of the project once, CPU only, 6 processes)
live2 found one race (2026-07-12 R10) with no race clock on screen, where the live runner said nothing. This counts how often that
happens over all downloaded days, with the same reader (identifier/live_run.Clock). Nothing else is read from the pictures."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "identifier"))


def scan(path):
    import live_run as LR
    cap = cv2.VideoCapture(str(path))
    clock, frames, on = LR.Clock(), 0, 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) != (1920, 1080):
        return {"video": str(path.relative_to(ROOT.parent)), "size": [w, h], "status": "not 1920x1080"}
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        v = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frames += 1
        on += bool(clock.feed(v, bgr))
    cap.release()
    st = "start and finish found" if clock.finish is not None else "start only" if clock.start is not None else "pill seen, never ran" if on else "no clock on screen"
    return {"video": str(path.relative_to(ROOT.parent)), "frames": frames, "pill_frames": on, "start": clock.start, "finish": clock.finish,
            "run": None if clock.finish is None else round(clock.finish - clock.start, 2), "status": st}


if __name__ == "__main__":
    vids = sorted(ROOT.glob("otherday*/videos/*.mp4")) + sorted((ROOT.parent / "onair_overnight_20261117" / "videos").glob("*.mp4"))
    with Pool(6) as p:
        rows = p.map(scan, vids, chunksize=1)
    count = {}
    for r in rows:
        count[r["status"]] = count.get(r["status"], 0) + 1
    by_day = {}
    for r in rows:
        d = r["video"].split("/")[0] if "otherday" in r["video"] else "onair_overnight_20261117"
        by_day.setdefault(d, {}).setdefault(r["status"], 0)
        by_day[d][r["status"]] += 1
    Path(sys.argv[1]).write_text(json.dumps({"videos": len(rows), "count": count, "by_folder": by_day, "rows": rows}, ensure_ascii=False, indent=1))
    print(len(rows), count)
    for r in rows:
        if r["status"] != "start and finish found":
            print(r)
