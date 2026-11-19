"""bench2, check 3: contact sheets of official-track input pictures, to look for graphics outside the strip box that give the order away.
usage: bw3_sheet.py BENCH_DIR OUT_DIR        (opencv; picks, with a fixed seed, test items of each kind and tiles the third picture of each)"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

B, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
ans = {x["id"]: x for x in map(json.loads, (B / "answers_official.jsonl").read_text().splitlines())}
items = [x for x in map(json.loads, (B / "items_official.jsonl").read_text().splitlines()) if x["split"] == "test"]
rng = random.Random(20261119)
kinds = {"start": [i for i in items if ans[i["id"]]["part"] == "start"],
         "finish_itself": [i for i in items if ans[i["id"]]["to_finish"] == 0],
         "last_3_to_12": [i for i in items if 3 <= ans[i["id"]]["to_finish"] <= 12],
         "last_15_to_36": [i for i in items if 15 <= ans[i["id"]]["to_finish"] <= 36]}
for kind, pool in kinds.items():
    pick = rng.sample(pool, min(12, len(pool)))
    tiles = []
    for it in pick:
        im = cv2.resize(cv2.imread(str(B / "inputs" / f"{it['id']}_3.jpg")), (640, 360))
        a = ans[it["id"]]
        cv2.putText(im, f"{it['id']}  to finish {a['to_finish']}s  top3 {a['top3']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(im)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)])
    cv2.imwrite(str(OUT / f"{kind}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(kind, len(pool), "items in the pool,", len(pick), "shown")
