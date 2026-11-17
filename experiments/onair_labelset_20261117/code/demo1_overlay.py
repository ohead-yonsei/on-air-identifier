"""Draw what identify_l1.py said on each time point of a video, and score it against the strip where the strip is on screen.
usage: demo1_overlay.py VIDEO RESULT.json OUT_DIR
Writes OUT_DIR/overlay_XXX.jpg (960 px wide) and OUT_DIR/summary.json.
Box colours: blue = taken as a front runner in this picture, green = number sure but not taken, grey = number not sure."""
import bisect
import json
import sys
from pathlib import Path

import cv2

video, result, out = sys.argv[1], json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
pts = json.loads(Path(sys.argv[2]).with_suffix(".pts.json").read_text())
cap = cv2.VideoCapture(video)
BLUE, GREEN, GREY = (235, 120, 30), (60, 170, 60), (150, 150, 150)
points, st = [], {"validated_points": 0, "statements": 0, "right": 0, "points_with_right": 0, "points_silent": 0}
for k, p in enumerate(result["points"]):
    i = bisect.bisect_right(pts, p["t"] + 1e-8) - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, bgr = cap.read()
    for r in p["riders_now"]:
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        col = BLUE if r.get("taken") else GREEN if r["sure"] else GREY
        cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 4 if r.get("taken") else 2)
        label = f"{r['number']}" + ("" if r["sure"] else "?")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cv2.rectangle(bgr, (x1, y1 - th - 14), (x1 + tw + 10, y1), col, -1)
        cv2.putText(bgr, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    name = f"overlay_{k:03d}.jpg"
    cv2.imwrite(str(out / name), cv2.resize(bgr, (960, 540), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
    strip = p.get("strip")
    row = {"k": k, "t": p["t"], "image": name, "statements": p["statements"], "detail": p["detail"], "strip": strip, "strip_on_screen": bool(strip),
           "riders": len(p["riders_now"]), "sure": sum(r["sure"] for r in p["riders_now"]), "detections": [x["detection"] for x in p["pictures"]]}
    if strip:
        row["right"] = [n for n in p["statements"] if n in strip[:3]]
        row["in_places_4_5"] = [n for n in p["statements"] if n in strip[3:]]
        row["not_on_strip"] = [n for n in p["statements"] if n not in strip]
        st["validated_points"] += 1
        st["statements"] += len(p["statements"])
        st["right"] += len(row["right"])
        st["points_with_right"] += bool(row["right"])
        st["points_silent"] += not p["statements"]
    points.append(row)
cap.release()
st["A"] = st["right"] / st["statements"] if st["statements"] else None
st["B"] = st["points_with_right"] / st["validated_points"] if st["validated_points"] else None
(out / "summary.json").write_text(json.dumps({"video": video, "live_timer_segment": result.get("live_timer_segment"), "start": result.get("start"), "end": result.get("end"),
                                              "seconds": result.get("seconds"), "qwen_failures": result.get("qwen_failures"), "score_on_strip_points": st, "points": points}, indent=1))
print("time points", len(points), "| with strip", st["validated_points"], "| statements", st["statements"], "right", st["right"], "| A", st["A"], "B", st["B"])
