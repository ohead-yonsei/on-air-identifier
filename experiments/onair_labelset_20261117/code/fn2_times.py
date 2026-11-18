"""Finish stretch 2, step 2: when the race clock on screen starts and stops, for every race video of a day.
usage: fn2_times.py DAY_DIR OUT.json        (DAY_DIR has videos/R??_*.mp4)

The whole video is decoded once. For every frame: is the clock graphic on screen (blue box + white digits), and how much the white
digit pixels inside the blue pill changed from the frame before (the first version looked at the whole graphic, whose surroundings
show the moving picture, and was off by up to a few seconds). S = first change of the digits in the longest run with the clock on screen, F = last change.
Between S and F the digits change every 0.1 s, so the race time at video time v is taken as  T_win - (F - v)  with the official winning time."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TIMER = (70, 930, 360, 1055)
DIGITS = (118, 968, 302, 1012)      # inside the blue pill that holds the digits (1920x1080)


def series(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    on, ch, prev = [], [], None
    x1, y1, x2, y2 = TIMER
    dx1, dy1, dx2, dy2 = DIGITS
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        blue = float(((h >= 100) & (h <= 125) & (s > 140) & (v > 90)).mean())
        w = (s < 60) & (v > 200)
        on.append(blue >= 0.06 and float(w.mean()) >= 0.10)
        d = cv2.cvtColor(bgr[dy1:dy2, dx1:dx2], cv2.COLOR_BGR2HSV)
        dw = (d[..., 1] < 90) & (d[..., 2] > 170)          # white digit pixels inside the pill; the background never shows here
        ch.append(0.0 if prev is None else float((dw != prev).mean()))
        prev = dw
    cap.release()
    return fps, np.array(on), np.array(ch)


def main():
    day, out = ROOT / sys.argv[1], Path(sys.argv[2])
    res = {}
    for path in sorted((day / "videos").glob("R??*.mp4")):
        race = path.name[:3]
        fps, on, ch = series(path)
        runs, start = [], None            # runs of frames with the clock on screen, gaps up to 3 s bridged
        idx = np.where(on)[0]
        if len(idx) == 0:
            res[race] = {"error": "no clock"}
            print(race, "no clock", flush=True)
            continue
        cur = [idx[0], idx[0]]
        for i in idx[1:]:
            if i - cur[1] > 3 * fps:
                runs.append(cur)
                cur = [i, i]
            cur[1] = i
        runs.append(cur)
        a, b = max(runs, key=lambda r: r[1] - r[0])
        moving = np.where((ch[a:b + 1] > 0.01) & on[a:b + 1])[0] + a
        # the digits tick every 0.1 s between S and F: keep the changes that have another change within 0.35 s
        tick = [i for k, i in enumerate(moving) if (k > 0 and i - moving[k - 1] <= 0.35 * fps) or (k + 1 < len(moving) and moving[k + 1] - i <= 0.35 * fps)]
        if len(tick) < 50:
            res[race] = {"error": "clock never runs", "clock_on": [round(float(a / fps), 2), round(float(b / fps), 2)]}
            print(race, "clock never runs", flush=True)
            continue
        # longest stretch of ticking without a pause of 1 s or more
        best, s0 = (tick[0], tick[0]), tick[0]
        for k in range(1, len(tick)):
            if tick[k] - tick[k - 1] > 1.0 * fps:
                s0 = tick[k]
            if tick[k] - s0 > best[1] - best[0]:
                best = (s0, tick[k])
        S, F = best
        res[race] = {"file": path.name, "fps": fps, "frames": int(len(on)), "clock_on": [round(float(a / fps), 3), round(float(b / fps), 3)],
                     "S": round(float(S / fps), 3), "F": round(float(F / fps), 3), "running": round(float((F - S) / fps), 3),
                     "clock_on_after_F": round(float((b - F) / fps), 2), "ticks": int(sum(1 for i in tick if S <= i <= F))}
        print(race, res[race], flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
