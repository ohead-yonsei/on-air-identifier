"""far (2026-11-20): cap labels for the far home-straight shots, from the official winner, without using what the classifier reads.
usage: x1_mine_front.py OUT_DIR DAY [DAY ...]        (CPU; writes OUT_DIR/mined.json and OUT_DIR/crops/, in the format h3_train_caps.py reads)

Rule: in a race whose clock agrees with the official time and whose winner was MARGIN s or more clear of the second, take the pictures
from T_FAR to T_NEAR seconds before the finish (before the winning post hides the leader). When the rider furthest right is GAP rider
heights or more ahead of the next rider, that rider is the winner; its cap picture gets the winner's number. At most PER pictures a race.
Earlier mining (h1, h2) could only label riders the classifier had NOT read; this one also labels riders it read as something else.
Pictures and crops stay on the server."""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import w4_winner_guard as w4
from h2_mine_caps_official import canvas, cls

MARGIN, T_NEAR, T_FAR, GAP, PER = 0.4, 0.8, 4.0, 2.5, 6


def main():
    out, days = ROOT / sys.argv[1], sys.argv[2:]
    (out / "crops").mkdir(parents=True, exist_ok=True)
    rows, races = [], Counter()
    for day in days:
        fc.FOLDER.setdefault(day, day)
        off, tms = fc.official(day), fc.times(day)
        for race, dr in sorted(w4.dense(day).items()):
            hs = sorted((h for h in off[race]["horses"] if h.get("rank") and h.get("time", {}).get("finish")), key=lambda h: h["rank"])
            races["all"] += 1
            if not dr["clock_agrees"] or len(hs) < 2 or hs[1]["time"]["finish"] - hs[0]["time"]["finish"] < MARGIN:
                continue
            races["clear winner"] += 1
            found = []
            for p in dr["pictures"]:
                if not (T_NEAR - 0.05 <= p["to_finish"] <= T_FAR + 0.05) or len(p["boxes"]) < 2:
                    continue
                b, P = np.array(p["boxes"], float), np.array(p["probs"], float)
                cx, h = (b[:, 0] + b[:, 2]) / 2, b[:, 3] - b[:, 1]
                o = np.argsort(-cx)
                i = int(o[0])
                if not (30 <= h[i] <= 130) or (cx[i] - cx[o[1]]) / h[i] < GAP:
                    continue
                c = int(np.argmax(P[i]))
                found.append({"day": day, "race": race, "frame": p["frame"], "to_finish": p["to_finish"], "rider": i, "box": p["boxes"][i], "number": hs[0]["number"],
                              "class": cls(hs[0]["number"]), "where": "front, clear winner", "read_as": c + 1, "read_p": round(float(P[i, c]), 4),
                              "gap": round(float((cx[i] - cx[o[1]]) / h[i]), 2), "height": round(float(h[i]), 1)})
            if not found:
                continue
            races["with labels"] += 1
            found.sort(key=lambda r: r["frame"])
            keep = [found[k] for k in sorted(set(int(round(k * (len(found) - 1) / (PER - 1))) for k in range(PER)))] if len(found) > PER else found
            want = {r["frame"]: r for r in keep}
            cap = cv2.VideoCapture(str(ROOT / fc.FOLDER[day] / "videos" / tms[race]["file"]))
            first, last = min(want), max(want)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            i = first
            while i <= last:
                ok, bgr = cap.read()
                if not ok:
                    break
                if i in want:
                    H, W = bgr.shape[:2]
                    b = bgr.copy()
                    b[955:1045, 480:1920] = 48
                    r = want[i]
                    r["crop"] = f"{day}_{race}_f{i}_{r['rider']}.png"
                    canvas(Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)), r["box"], W, H).save(out / "crops" / r["crop"])
                    rows.append(r)
                i += 1
            cap.release()
        print(day, "labels so far", len(rows), flush=True)
    by = Counter(r["class"] for r in rows)
    same = sum(r["read_as"] == r["class"] and r["read_p"] >= 0.7 for r in rows)
    other = Counter((r["class"], r["read_as"]) for r in rows if r["read_as"] != r["class"] and r["read_p"] >= 0.7)
    unsure = sum(r["read_p"] < 0.7 for r in rows)
    print("races", dict(races), "| labels", len(rows), "by class", sorted(by.items()))
    print(f"the classifier reads the label's number (0.7 or more): {same}; is sure of another number: {sum(other.values())}; is not sure: {unsure}")
    print("sure of another number (label, read):", sorted(other.items(), key=lambda kv: -kv[1])[:20])
    (out / "mined.json").write_text(json.dumps({"days": days, "rule": {"margin": MARGIN, "from": T_FAR, "to": T_NEAR, "gap": GAP, "per_race": PER}, "by_class": dict(sorted(by.items())),
                                                "reads_the_label": same, "sure_of_another_number": sum(other.values()), "not_sure": unsure, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
