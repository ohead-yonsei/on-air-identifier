"""Check of the program's --v10 option (identifier/identify_l1.py) against the experiment code and the official record.
usage: fn13_program_v10.py demo PROGRAM.json DEMO_DIR            the program on the script-demo race against code/ls2_run.py's ident.json
       fn13_program_v10.py days OUT.json DAY [DAY ...]            the program on whole days (finish2/program_v10/DAY/Rxx.json):
                                                                  leader in the last 36 s and winner call against the official record,
                                                                  and the winner call against the experiment's (finish2/winner_C_dense.json)
The days used here were already scored in finish2 (their results have been seen); this is a check of the program, not a new test."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc


def demo(program, demo_dir):
    new = json.loads(Path(program).read_text())
    old = json.loads((ROOT / demo_dir / "ident.json").read_text())
    n = same_s = same_l = late = same_late = 0
    for p in new["points"]:
        q = next((x for x in old["points"] if abs(x["t"] - p["t"]) < 0.05 and not x.get("early")), None)
        if q is None:
            continue
        n += 1
        same_s += p["statements"] == q["statements"]
        ok = p.get("leader") == q["leader"] and p.get("leader_source") == q["leader_source"]
        same_l += ok
        if q["late"]:
            late += 1
            same_late += ok
        if p["statements"] != q["statements"] or not ok:
            print("differs", p["t"], p["statements"], p.get("leader"), p.get("leader_source"), "| demo", q["statements"], q["leader"], q["leader_source"], "late", q["late"], p.get("late"))
    w, v = new["winner_call"], old["winner_call"]
    print(f"demo race: time points compared {n}; identical statements {same_s}, identical leader {same_l} (in the last stretch {same_late} of {late}); "
          f"winner call program {w['number']} {w['pictures_naming_each']} | demo {v['number']} {v['pictures_naming_each']}; "
          f"finish program {new['finish']['finish']} demo {old['clock']['stop']}")


def days(out, names):
    exp = {}
    f = ROOT / "finish2" / "winner_C_dense.json"
    if f.exists():
        for r in json.loads(f.read_text())["every_race_run"]["rows"]:
            exp[(r["day"], r["race"])] = r["call"]
    rows, tot = [], {k: {"races": 0, "called": 0, "right": 0, "same_as_experiment": 0, "points": 0, "said": 0, "right_strict": 0, "right_lenient": 0,
                         "strip_said": 0, "strip_right_strict": 0, "finish_off": []} for k in ("main", "every_race_run")}
    for day in names:
        off, tms = fc.official(day), fc.times(day)
        for f in sorted((ROOT / "finish2" / "program_v10" / day).glob("R??.json")):
            race, o = f.stem, json.loads(f.read_text())
            if race not in off or "winner_call" not in o:
                continue
            t_win = fc.winner_time(off[race])
            vf, how, agrees = fc.video_finish(tms.get(race), t_win)
            if vf is None:
                continue
            rank = {h["number"]: h["rank"] for h in off[race]["horses"]}
            call = o["winner_call"]["number"]
            row = {"day": day, "race": race, "clock_agrees": agrees, "call": call, "rank_of_call": rank.get(call), "experiment_call": exp.get((day, race), "absent"),
                   "finish_program": o["finish"]["finish"], "finish_reference": vf, "points": []}
            for p in o["points"]:
                tg = vf - p["t"]
                if not (0 <= tg <= 36) or not p.get("late"):
                    continue
                ref = fc.reference(off[race], t_win - tg)
                if ref is None:
                    continue
                row["points"].append({"to_finish": round(tg, 1), "leader": p.get("leader"), "strip_model": p.get("leader_strip_model"), "ref": ref["leader"], "ref_lenient": ref["leader_lenient"]})
            rows.append(row)
            for k in ("main", "every_race_run"):
                if k == "main" and not agrees:
                    continue
                s = tot[k]
                s["races"] += 1
                s["called"] += call is not None
                s["right"] += call is not None and rank.get(call) == 1
                s["same_as_experiment"] += row["experiment_call"] == call
                s["finish_off"].append(round(o["finish"]["finish"] - vf, 2))
                for q in row["points"]:
                    s["points"] += 1
                    s["said"] += q["leader"] is not None
                    s["right_strict"] += q["leader"] is not None and q["leader"] == q["ref"]
                    s["right_lenient"] += q["leader"] is not None and q["leader"] in q["ref_lenient"]
                    s["strip_said"] += q["strip_model"] is not None
                    s["strip_right_strict"] += q["strip_model"] is not None and q["strip_model"] == q["ref"]
    for k, s in tot.items():
        fo = sorted(abs(v) for v in s["finish_off"])
        pc = lambda a, b: f"{100 * a / b:.1f}%" if b else "-"
        print(f"{k}: races {s['races']} | winner call: called {s['called']}, right {s['right']}, same as the experiment's call in {s['same_as_experiment']} races"
              f" | leader in the last 36 s: said {s['said']} of {s['points']} time points, right {s['right_strict']} ({pc(s['right_strict'], s['said'])}), "
              f"within 3 m {s['right_lenient']} ({pc(s['right_lenient'], s['said'])}); strip-trained model on the same points: said {s['strip_said']}, right {s['strip_right_strict']} ({pc(s['strip_right_strict'], s['strip_said'])})"
              f" | finish time, program minus reference: median abs {fo[len(fo) // 2] if fo else None} s, max abs {fo[-1] if fo else None} s")
    Path(out).write_text(json.dumps({"days": names, "totals": tot, "rows": rows}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "demo":
        demo(sys.argv[2], sys.argv[3])
    else:
        days(sys.argv[2], sys.argv[3:])
