"""Finish stretch 2, stage 2: statements for the last 36 s from the dense pictures (fn6_dense.py), for one variant.
usage: fn7_method.py VARIANT DAY [DAY ...]

Writes finish2/runs/VARIANT/DAY.json in the format of fn3_run.py, so fn4_score.py scores it unchanged.
The two time points before the stretch (39 and 42 s to go), needed by the carry rule, come from the F2 run of the same race.

Variants (all without Qwen, all without the strip):
  P3    each picture on its own, pictures at t, t-0.6, t-1.0 s, stated when taken in 2 of 3   (the F2 rule on the dense pictures; check)
  P6    each picture on its own, six pictures of the last second, stated when taken in 3 of 6
  M3    riders followed across pictures (fn_tracks), number from the average along the track; 2 of 3
  M6    as M3 with six pictures, 3 of 6
  *f    the same with the order and leader models learned on the finish stretch from official times (finish2/models/fs_*.joblib;
        on development days the model that left that day out)
  *l    the same with only the leader model learned that way (statements as without the suffix)"""
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft

SIX = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
VARIANTS = {"P3": dict(memory=False, offsets=(0.0, 0.6, 1.0), need=2), "P6": dict(memory=False, offsets=SIX, need=3),
            "M3": dict(memory=True, offsets=(0.0, 0.6, 1.0), need=2), "M6": dict(memory=True, offsets=SIX, need=3),
            "M6n4": dict(memory=True, offsets=SIX, need=4), "P6n4": dict(memory=False, offsets=SIX, need=4),
            # exploration only: looser tracks, lower confidence needed after averaging
            "M3a": dict(memory=True, offsets=(0.0, 0.6, 1.0), need=2, sure=0.5), "M3b": dict(memory=True, offsets=(0.0, 0.6, 1.0), need=2, link=dict(gate=2.0, max_gap=5, min_matched=0.3)),
            "M3c": dict(memory=True, offsets=(0.0, 0.6, 1.0), need=2, sure=0.5, link=dict(gate=2.0, max_gap=5, min_matched=0.3)),
            "M6c": dict(memory=True, offsets=SIX, need=3, sure=0.5, link=dict(gate=2.0, max_gap=5, min_matched=0.3)),
            "P3a": dict(memory=False, offsets=(0.0, 0.6, 1.0), need=2, sure=0.5)}
DEV = ["otherday", "otherday2", "otherday3", "otherday4", "otherday5", "20261115"]


def models(variant, day):
    """Order model and leader model. Suffix f: both learned on the finish stretch; suffix l: only the leader model (the order model stays the strip-trained one)."""
    strip = (joblib.load(ROOT / "night" / "l1_fast.joblib"), joblib.load(ROOT / "night" / "leader_fast.joblib"))
    if variant[-1] not in "fl":
        return strip
    tag = f"without_{day}" if day in DEV else "all"
    fs = (joblib.load(ROOT / "finish2" / "models" / f"fs_order_{tag}.joblib"), joblib.load(ROOT / "finish2" / "models" / f"fs_leader_{tag}.joblib"))
    return fs if variant.endswith("f") else (strip[0], fs[1])


def run(variant, day):
    v = VARIANTS[variant.rstrip("fl")]
    order_model, leader_model = models(variant, day)
    dense = json.loads((ROOT / "finish2" / "dense" / f"{day}.json").read_text())["races"]
    base = json.loads((ROOT / "finish2" / "runs" / "F2" / f"{day}.json").read_text())["races"]
    res = {"day": day, "config": variant, "races": {}}
    for race, dr in sorted(dense.items()):
        pics = dr["pictures"]
        ids = ft.link(pics, **v.get("link", {}))
        who = ft.identify(pics, ids, memory=v["memory"])
        per = {round(p["to_finish"], 1): ft.decide(p, w, order_model, leader_model, sure=v.get("sure", ft.SURE)) for p, w in zip(pics, who)}
        hist, hist_lead = {}, {}
        for p in base.get(race, {}).get("points", []):
            if p["to_finish"] > 36:
                hist[p["to_finish"]] = {d["number"] for d in p["detail"] if d["source"] == "now"}
                hist_lead[p["to_finish"]] = p["leader"] if p.get("leader_source") == "now" else None
        points = ft.statements(per, list(range(36, -1, -3)), hist, hist_lead, offsets=v["offsets"], need=v["need"])
        res["races"][race] = {"video_finish": dr["video_finish"], "anchor": dr["anchor"], "clock_agrees": dr["clock_agrees"], "winner_time": dr["winner_time"],
                              "points": points, "tracks": max((max(i) for i in ids if i), default=-1) + 1}
    out = ROOT / "finish2" / "runs" / variant / f"{day}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(variant, day, len(res["races"]), "races", flush=True)


if __name__ == "__main__":
    for d in sys.argv[2:]:
        run(sys.argv[1], d)
