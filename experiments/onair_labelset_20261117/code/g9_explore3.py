"""finish3 stage 2, third exploration (days seen before): the true leader is not "readable" at about 30 % of the time points. Why, and does a
lower bar for a rider's number help the leader statement?
usage: g9_explore3.py OUT.json            (CPU only; HELD days = the six measurement and confirmation days of finish2, models as in the base v10)

1. At the time points where the true leader is not readable (not a sure rider in 2 of the 3 pictures): how many riders were found, how many
   of them had a sure number, against the size of the field. Few riders found = the detector is the limit; many found but few sure = the cap
   reading is the limit. (Which box is the true leader is not known, so this is indirect.)
2. v10's leader with the bar for "sure" lowered from 0.7 to 0.6 / 0.5 / 0.4 (only for the leader; the models are the same ones)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft
import g5_explore as g5
import g8_explore2 as g8


def main():
    out, res = Path(sys.argv[1]), {}
    strip, fs_all = joblib.load(ROOT / "night" / "leader_fast.joblib"), joblib.load(ROOT / "finish2" / "models" / "fs_leader_all.joblib")
    runs = {}
    for sure in (0.7, 0.6, 0.5, 0.4):
        ft.SURE = sure
        data = {d: g5.load(d) for d in g5.HELD}
        for d in g5.HELD:
            g5.score_day(data[d], {"strip": (strip, 19), "fs": (fs_all, 19), "W19": (strip, 19)})
        runs[sure] = {(d, race): g8.race_rows(rr, ("thr", 0.5)) for d in g5.HELD for race, rr in data[d].items() if rr["agree"]}
        if sure == 0.7:
            # detection statistics at the time points, from the raw dense file
            stat = {"readable": [], "unreadable": []}
            for d in g5.HELD:
                dense = json.loads((ROOT / "finish3" / "dense" / f"{d}.json").read_text())["races"]
                off = fc.official(d)
                for race, rr in data[d].items():
                    if not rr["agree"]:
                        continue
                    field = len(off[race]["horses"])
                    by_tau = {p["to_finish"]: p for p in dense[race]["pictures"]}
                    for r in runs[sure][(d, race)]:
                        p = by_tau.get(float(r["tau"]))
                        if p is None:
                            continue
                        n_sure = sum(1 for row in p["probs"] if max(row) >= 0.7) if p["boxes"] else 0
                        stat["readable" if r["readable"] else "unreadable"].append((len(p["boxes"]), n_sure, field, r["part"]))
            for k, v in stat.items():
                a = np.array([x[:3] for x in v], float)
                res[f"riders_{k}"] = {"points": len(v), "found_mean": round(float(a[:, 0].mean()), 2), "sure_mean": round(float(a[:, 1].mean()), 2), "field_mean": round(float(a[:, 2].mean()), 2),
                                      "found_under_half_field": round(float((a[:, 0] < a[:, 2] / 2).mean() * 100), 1), "no_sure_rider": round(float((a[:, 1] == 0).mean() * 100), 1),
                                      "one_or_two_sure": round(float(((a[:, 1] >= 1) & (a[:, 1] <= 2)).mean() * 100), 1)}
                print(k, res[f"riders_{k}"])
    base = runs[0.7]
    for sure, by in runs.items():
        rows = [r for v in by.values() for r in v]
        t = g5.table(by)
        w = t["whole race"]
        extra = {} if sure == 0.7 else {"right_share_minus_base": g5.paired(by, base), "wrong_share_minus_base": g5.paired(by, base, "wrong_share")}
        res[f"sure {sure}"] = {"parts": t, "readable_share": round(100 * sum(r["readable"] for r in rows) / len(rows), 1), **extra}
        print(f"sure {sure}: true leader readable {res[f'sure {sure}']['readable_share']}% | whole race right share {w['right_share']}%  wrong share {w['wrong_share']}%  when said {w['when_said']}% ({w['right']}/{w['said']})"
              f" | start {t['start']['right_share']}/{t['start']['wrong_share']}  middle {t['middle']['right_share']}/{t['middle']['wrong_share']}  finish {t['finish']['right_share']}/{t['finish']['wrong_share']}"
              + ("" if not extra else f" | minus base: right {extra['right_share_minus_base']} wrong {extra['wrong_share_minus_base']}"))
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
