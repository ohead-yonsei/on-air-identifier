"""winner (2026-11-20): what the winner call does picture by picture in the last seconds, with the v11 cap classifier.
usage: w1_winner_look.py OUT.json DAY [DAY ...]        (CPU only; reads caps2/dense/A/last/DAY.json = the riders of the last 37 s, 5 pictures a second)

Per race: the current rule's call (most-named leader pick in the last 6 s, 5 pictures or more), the official winner and second, and a
timeline of the last 6 s: per picture the leader pick, whether the winner's number is a sure rider, how many riders were found.
Exploration on days already seen; the answer (official placing) is used only to score."""
import json
import sys
from collections import Counter
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft

ORDER = joblib.load(ROOT / "night" / "l1_fast.joblib")
STRIP = joblib.load(ROOT / "night" / "leader_fast.joblib")
FS = joblib.load(ROOT / "finish2" / "models" / "fs_leader_all.joblib")


def race_pictures(day, name="A"):
    f = ROOT / "caps2" / "dense" / name / "last" / f"{day}.json"
    fc.FOLDER.setdefault(day, day)
    off = fc.official(day)
    for race, dr in sorted(json.loads(f.read_text())["races"].items()):
        pics = [p for p in dr["pictures"] if -0.05 <= p["to_finish"] <= 8.05]
        who = ft.identify(pics, ft.link(pics), memory=False)
        rank = {h["number"]: h["rank"] for h in off[race]["horses"] if h.get("rank")}
        rows = []
        for p, w in zip(pics, who):
            d, dfs = ft.decide(p, w, ORDER, STRIP), ft.decide(p, w, ORDER, FS)
            hs = [b[3] - b[1] for b in p["boxes"]]
            rows.append({"tf": round(p["to_finish"], 1), "riders": len(p["boxes"]), "detection": p["detection"], "h_max": round(max(hs), 1) if hs else 0,
                         "who": [(n, round(c, 3)) for n, c in w], "boxes": p["boxes"], "pick": d["pick"], "first": {k: round(v, 3) for k, v in d["first"].items()},
                         "pick_fs": dfs["pick"], "first_fs": {k: round(v, 3) for k, v in dfs["first"].items()}})
        yield race, dr["clock_agrees"], rank, rows


def main():
    out_path, days = sys.argv[1], sys.argv[2:]
    res, tot = {}, Counter()
    for day in days:
        for race, agree, rank, rows in race_pictures(day):
            win = next((n for n, r in rank.items() if r == 1), None)
            second = next((n for n, r in rank.items() if r == 2), None)
            c = Counter(r["pick"] for r in rows if r["pick"] is not None and r["tf"] <= 6.05)
            top = c.most_common(1)
            call = top[0][0] if top and top[0][1] >= 5 else None
            kind = "none" if call is None else ("right" if call == win else "wrong")
            tot[kind] += 1
            tot[("agree" if agree else "no clock", kind)] += 1
            wcap = min(win, 11) if win else None
            line = ""
            for r in sorted((r for r in rows if r["tf"] <= 6.05), key=lambda r: -r["tf"]):
                sure_w = any(n == wcap and p >= 0.7 for n, p in r["who"])
                line += (str(r["pick"] % 10) if r["pick"] is not None else ("w" if sure_w else ("-" if r["riders"] else "_")))
            res[f"{day}/{race}"] = {"agree": agree, "winner": win, "second": second, "call": call, "kind": kind, "counts": dict(c), "rows": rows}
            if kind != "right":
                print(f"{day:10s} {race} {'  ' if agree else 'nc'} win {win:2d} 2nd {second:2d} call {str(call):4s} {kind:5s} {dict(c.most_common(3))}  {line}")
    print({k: v for k, v in tot.items() if isinstance(k, str)}, {f"{a} {b}": v for (a, b), v in ((k, v) for k, v in tot.items() if isinstance(k, tuple))})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
