"""ON-AIR draft benchmark v0: make the pictures a system under test gets.
usage: python make_inputs.py --exp EXPERIMENT_DIR [--split test] [--out inputs] [--items items_official.jsonl]        (needs opencv; run where the videos are)

Per item three pictures: 1.0 s before, 0.5 s before, and at the moment of the item (1920x1080 JPEG, quality 90),
with the running-order strip painted over in grey (races.json "strip_box"). Nothing else is hidden: the race clock and the
race information graphic stay visible. Files: inputs/ID_1.jpg (earliest), ID_2.jpg, ID_3.jpg (the moment itself)."""
import argparse
import bisect
import json
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--split", default="all")
ap.add_argument("--out", default=str(HERE / "inputs"))
ap.add_argument("--items", default="items.jsonl", help="items.jsonl (strip track) or items_official.jsonl (official track)")
a = ap.parse_args()
EXP, OUT = Path(a.exp), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
meta = json.loads((HERE / "races.json").read_text())
x1, y1, x2, y2 = meta["strip_box"]
items = [json.loads(l) for l in (HERE / a.items).read_text().splitlines()]
by_race = {}
for it in items:
    if a.split in ("all", it["split"]):
        by_race.setdefault(it["race_key"], []).append(it)
done = 0
for key, its in sorted(by_race.items()):
    r = meta["races"][key]
    folder, race = EXP / r["source_folder"], f"R{r['race']:02d}"
    video = next((folder / "videos").glob(f"{race}_*.mp4"))
    pts = json.loads((folder / "evidence" / f"{race}_pts.json").read_text())
    cap = cv2.VideoCapture(str(video))
    for it in its:
        for k, t in enumerate(it["picture_times"], 1):
            f = OUT / f"{it['id']}_{k}.jpg"
            if f.exists():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, bisect.bisect_right(pts, t + 1e-8) - 1)
            ok, bgr = cap.read()
            assert ok and bgr.shape[:2] == (1080, 1920), (it["id"], t)
            bgr[y1:y2, x1:x2] = 128
            cv2.imwrite(str(f), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        done += 1
    cap.release()
    print(key, len(its), "items", flush=True)
print("items", done)
