"""ON-AIR draft benchmark v0: build the items, the answers and the race cards from the experiment records.
usage: python3 benchmark/v0/build_items.py        (from the project root; standard library only)

An item = one moment of a race at which the broadcaster's running-order strip is on screen.
Answer   = the five numbers on the strip at that moment, first to fifth (read from the broadcast by a local model, see README).
Writes items.jsonl (what a system under test may see), answers.jsonl (kept apart), races.json (race cards)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent / "experiments" / "onair_labelset_20261117"
# test = the six race days that were never used to train or choose anything in the identifier; dev = the other five
DAYS = [("otherday", "dev"), ("otherday2", "dev"), ("otherday3", "dev"), ("otherday4", "dev"), ("otherday5", "dev"),
        ("otherday6", "test"), ("otherday7", "test"), ("otherday8", "test"), ("otherday9", "test"), ("otherday10", "test"), ("otherday11", "test")]
STRIP_BOX = [480, 955, 1920, 1045]      # x1, y1, x2, y2 at 1920x1080: the strip; hidden in the pictures a system gets

items, answers, races = [], [], {}
for folder, split in DAYS:
    man = json.loads((EXP / folder / "manifest.json").read_text())
    src = {s["race"]: s for s in json.loads((EXP / folder / "video_sources.json").read_text())}
    off = json.loads((EXP / "finish2" / "official" / f"{folder}.json").read_text())["races"]
    tms = json.loads((EXP / "finish2" / "times" / f"{folder}.json").read_text())
    date = man["day"]
    past = {}
    for p in man["past"]:
        past.setdefault(p["frame"], {})[p["name"][-3:]] = p["pts"]
    for f in man["frames"]:
        if not f.get("strip"):
            continue
        race, key = f["race"], f"{date.replace('-', '')}_{f['race']}"
        o = off.get(race, {})
        if key not in races:
            ran = sorted(h["number"] for h in o.get("horses", []) if (h.get("time") or {}).get("S1F") is not None or h.get("rank") is not None)
            races[key] = {"date": date, "place": "Seoul", "race": int(race[1:]), "distance_m": o.get("distance"), "runners": ran,
                          "video": {"youtube_id": src[race]["id"], "title": src[race]["title"], "channel": src[race]["channel"],
                                    "width": src[race]["width"], "height": src[race]["height"], "fps": src[race]["fps"]},
                          "split": split, "source_folder": folder}
        t = tms.get(race, {})
        elapsed = round(f["pts"] - t["S"] + 0.1, 1) if "S" in t else None      # the clock shows 0:00.1 at S
        iid = f"{key}_t{int(round(f['pts'])):03d}"
        items.append({"id": iid, "race_key": key, "split": split, "video_time": f["pts"], "frame_index": f["decoded_index"],
                      "picture_times": [past[f["frame"]]["m10"], past[f["frame"]]["m05"], f["pts"]],
                      "seconds_since_start": elapsed if elapsed is not None and elapsed > 0 else None})
        answers.append({"id": iid, "strip": f["strip"]})

(HERE / "items.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
(HERE / "answers.jsonl").write_text("".join(json.dumps(x) + "\n" for x in answers))
(HERE / "races.json").write_text(json.dumps({"strip_box": STRIP_BOX, "races": races}, ensure_ascii=False, indent=1))
for s in ("dev", "test"):
    r = [k for k, v in races.items() if v["split"] == s]
    print(s, "days", len({races[k]["date"] for k in r}), "races", len(r), "items", sum(i["split"] == s for i in items))
