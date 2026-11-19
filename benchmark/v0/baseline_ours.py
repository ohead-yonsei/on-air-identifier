"""ON-AIR draft benchmark v0: baseline "our flow" = identifier, then a fixed sentence pattern.
usage: python3 baseline_ours.py IDENTIFIER_OUTPUT.json [...] --name NAME [--official]        (standard library only)

IDENTIFIER_OUTPUT.json: what identifier/identify_l1.py --day-dir DAY writes (one per race day; it never reads the strip to decide).
With --official (the official track, items_official.jsonl): the whole-race output of the same identifier class, every 3 s back from the
finish (code/fn10_whole.py for v9, code/bw1_whole.py for v11); an item is matched by race and video time.
The flow: where the identifier is sure, say the leading group (and the leader when it names one); where it is not, say
confirmed information only (place, race number, distance, seconds since the start). No language model is used here, so the
sentences are plain; the point is what is claimed, not how it is worded.
Writes baselines/NAME.jsonl."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("outputs", nargs="+")
ap.add_argument("--name", required=True)
ap.add_argument("--official", action="store_true")
a = ap.parse_args()
meta = json.loads((HERE / "races.json").read_text())["races"]
items = {}
for l in (HERE / ("items_official.jsonl" if a.official else "items.jsonl")).read_text().splitlines():
    it = json.loads(l)
    r = meta[it["race_key"]]
    if a.official:
        items[(r["source_folder"], f"R{r['race']:02d}", round(it["video_time"], 3))] = it      # (day folder, race, video time)
    else:
        items[(r["source_folder"], it["id"].split("_", 1)[1])] = it          # (day folder, frame name such as R02_t025)
lines = []
for path in a.outputs:
    res = json.loads(Path(path).read_text())
    if a.official:
        points = [((res["day"], race, round(p["t"], 3)), p) for race, rr in res["races"].items() for p in rr.get("points", [])]
    else:
        points = [((res["day_dir"], p["frame"]), p) for p in res["points"]]
    for k, p in points:
        it = items.get(k)
        if it is None:
            continue
        r = meta[it["race_key"]]
        said, lead = list(p["statements"]), p.get("leader")
        if lead is not None:
            others = [n for n in said if n != lead]
            text = f"선두는 {lead}번" + (", 선두권에 " + ", ".join(f"{n}번" for n in others) + "도 있습니다." if others else "입니다.")
        elif said:
            text = "선두권에 " + ", ".join(f"{n}번" for n in said) + "이 달리고 있습니다."
        else:
            text = f"서울 {r['race']}경주 {r['distance_m']}m, 출발 후 {int(it['seconds_since_start'])}초가 지났습니다."
        lines.append({"id": it["id"], "text": text, "mode": "ranking" if (said or lead is not None) else "confirmed information"})
out = HERE / "baselines" / f"{a.name}.jsonl"
out.parent.mkdir(exist_ok=True)
out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
print(a.name, len(lines), "lines;", sum(x["mode"] == "ranking" for x in lines), "with a ranking")
