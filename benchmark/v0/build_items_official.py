"""ON-AIR draft benchmark v0, official track: items at moments the running-order strip does not cover (the first seconds, the last
half minute, the finish itself), with the answer from the official passing times.
usage: python3 build_items_official.py --exp EXPERIMENT_DIR        (standard library only; needs EXP/<day>/evidence/*_pts.json, so run where the videos are)

Races: the races of races.json whose on-screen clock agrees with the official winning time (the video time of the finish is then known
to about 0.2 s). Time points: the winner's finish and every 3 s before it, back to 9 s after the start (as code/fn10_whole.py).
A time point becomes an item when no strip item of the same race lies within 1.5 s of it.
Answer: the order from the official passing times of every horse (code/fn_common.reference_full): first three, leader, and the lenient
sets (horses within 3 m of the third / of the first, because the sheet gives times to 0.1 s = about 1.7 m). At the finish itself the
answer is the official placing.
Writes items_official.jsonl (what a system may see), answers_official.jsonl (kept apart), official_races.json (which races, and why not)."""
import argparse
import bisect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
a = ap.parse_args()
EXP = Path(a.exp)
sys.path.insert(0, str(EXP / "code"))
import fn_common as fc          # noqa: E402

meta = json.loads((HERE / "races.json").read_text())["races"]
strip = {}
for l in (HERE / "items.jsonl").read_text().splitlines():
    it = json.loads(l)
    strip.setdefault(it["race_key"], []).append(it["video_time"])
items, answers, log = [], [], {}
for key, r in sorted(meta.items()):
    day, race = r["source_folder"], f"R{r['race']:02d}"
    off, tm = fc.official(day).get(race, {}), fc.times(day).get(race)
    tw = fc.winner_time(off) if "horses" in off else None
    vf, how, agree = fc.video_finish(tm, tw)
    if vf is None or not agree:
        log[key] = {"used": False, "why": how if vf is None else f"clock does not agree with the official time ({how})"}
        continue
    pts = json.loads((EXP / day / "evidence" / f"{race}_pts.json").read_text())
    rank = {h["number"]: h["rank"] for h in off["horses"] if h.get("rank")}
    n_items = 0
    for tau in range(3 * int((tw - 9.0) // 3), -1, -3):
        i = bisect.bisect_right(pts, vf - tau + 1e-8) - 1
        if i < 0 or pts[i] - 1.0 < pts[0]:
            continue
        t, rt = pts[i], round(tw - tau, 1)
        if any(abs(t - s) < 1.5 for s in strip.get(key, [])):
            continue
        ref = fc.reference_full(off, rt)
        if ref is None:
            continue
        if tau == 0:            # the finish itself: the official placing
            placed = sorted((n for n in ref["order"] if n in rank), key=rank.get)
            ref["order"] = placed + [n for n in ref["order"] if n not in rank]
            ref["top3"], ref["leader"] = ref["order"][:3], ref["order"][0]
            ref["top3_lenient"] = sorted(set(ref["top3_lenient"]) | set(ref["top3"]), key=ref["order"].index)
            ref["leader_lenient"] = sorted(set(ref["leader_lenient"]) | {ref["leader"]}, key=ref["order"].index)
        iid = f"{key}_n{int(round(t)):03d}"
        items.append({"id": iid, "race_key": key, "split": r["split"], "video_time": t, "picture_times": [round(t - 1.0, 3), round(t - 0.5, 3), t],
                      "seconds_since_start": rt, "track": "official"})
        answers.append({"id": iid, "to_finish": tau, "race_time": rt, "part": "start" if rt < 18 else ("finish" if tau <= 36 else "middle"),
                        "order": ref["order"], "top3": ref["top3"], "leader": ref["leader"], "top3_lenient": ref["top3_lenient"], "leader_lenient": ref["leader_lenient"]})
        n_items += 1
    log[key] = {"used": True, "items": n_items, "video_finish": vf, "winner_time": tw}
assert len({x["id"] for x in items}) == len(items)
(HERE / "items_official.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
(HERE / "answers_official.jsonl").write_text("".join(json.dumps(x) + "\n" for x in answers))
(HERE / "official_races.json").write_text(json.dumps(log, indent=1))
part = {x["id"]: x for x in answers}
for s in ("dev", "test"):
    its = [x for x in items if x["split"] == s]
    used = [k for k, v in log.items() if v["used"] and meta[k]["split"] == s]
    print(s, "races", len(used), "of", sum(v["split"] == s for v in meta.values()), "items", len(its),
          {p: sum(part[x["id"]]["part"] == p for x in its) for p in ("start", "middle", "finish")},
          "finish itself", sum(part[x["id"]]["to_finish"] == 0 for x in its))
