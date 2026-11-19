import json, sys
import cv2, numpy as np
d = json.load(open("far/mine/mined.json"))
kind = sys.argv[1]
rows = [r for r in d["rows"] if (r["read_as"] != r["class"] and r["read_p"] >= 0.7) == (kind == "other") and (kind == "other" or r["read_p"] < 0.7)]
rows.sort(key=lambda r: (r["class"], r["day"], r["race"], r["frame"]))
tiles = []
for r in rows[:60]:
    im = cv2.resize(cv2.imread("far/mine/crops/" + r["crop"]), (176, 176), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(im, (0, 0), (175, 22), (0, 0, 0), -1)
    cv2.putText(im, "L%d r%d %.2f %s%s" % (r["class"], r["read_as"], r["read_p"], r["day"][-2:], r["race"]), (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    tiles.append(im)
while len(tiles) % 10: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2], np.vstack([np.hstack(tiles[k:k + 10]) for k in range(0, len(tiles), 10)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(len(rows), "rows")
