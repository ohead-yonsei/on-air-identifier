"""winner: the winner call with a "still in front at the end" condition, scored against the official winner.
usage: w4_winner_guard.py score OUT.json DAY [DAY ...]      per race and in total; prints the verdicts of winner/DESIGN_KO.md
       w4_winner_guard.py same DAY                          do winner/dense/DAY.json and caps2/dense/A/last/DAY.json give the same calls?
       w4_winner_guard.py calls OUT.json DAY [DAY ...]      the calls only (for the benchmark flow)

Pictures: the riders of the last seconds, 5 pictures a second (winner/dense/DAY.json from w5_dense_last.py, else caps2/dense/A/last/DAY.json).
The call (unchanged from v10, identify_l1.py winner_call): in the 31 pictures of the last 6 s each picture names the strip-trained
leader model's pick among the sure riders (cap probability 0.7 or more, model score 0.5 or more); the number named most often is the
candidate when it was named in 5 pictures or more.
The condition (new): in the pictures of the last GUARD_T seconds, the rider that looks most like the candidate's cap "is seen" when
that probability is 0.5 or more, and "is in front" when no other rider's centre is more than 0.3 of its height further right (the
home straight in Seoul is filmed from the stands; the field runs left to right). The call stands when the candidate is seen in at
least GUARD_SEEN pictures and is in front in at least half of the pictures that see it. Otherwise there is no call.
The official winner is used only to score."""
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import fn_tracks as ft

ORDER = joblib.load(ROOT / "night" / "l1_fast.joblib")
STRIP = joblib.load(ROOT / "night" / "leader_fast.joblib")
SPAN, NEED = 6.0, 5                                  # the call, as v10
GUARD_T, GUARD_SEEN, GUARD_SHARE = 2.0, 1, 0.5       # the condition (primary rule)
SEE, AHEAD = 0.5, 0.3
SECOND = (3.0, 2, 0.5)                               # a second setting, reported only


def dense(day, source=None):
    a, b = ROOT / "winner" / "dense" / f"{day}.json", ROOT / "caps2" / "dense" / "A" / "last" / f"{day}.json"
    f = {"winner": a, "caps2": b}[source] if source else (a if a.exists() else b)
    return json.loads(f.read_text())["races"]


def vote(pics):
    who = ft.identify(pics, ft.link(pics), memory=False)
    c = Counter()
    for p, w in zip(pics, who):
        if -0.05 <= p["to_finish"] <= SPAN + 0.05:
            n = ft.decide(p, w, ORDER, STRIP)["pick"]
            if n is not None:
                c[n] += 1
    top = c.most_common(1)
    return (top[0][0] if top and top[0][1] >= NEED else None), dict(c)


def in_front(pics, n, T):
    """(pictures of the last T seconds that see number n's cap, pictures among them where it is in front)"""
    cap, seen, front = min(n, 11) - 1, 0, 0
    for p in pics:
        if not (-0.05 <= p["to_finish"] <= T + 0.05) or not p["boxes"]:
            continue
        b, P = np.array(p["boxes"], float), np.array(p["probs"], float)
        cx, h = (b[:, 0] + b[:, 2]) / 2, b[:, 3] - b[:, 1]
        i = int(np.argmax(P[:, cap]))
        if P[i, cap] < SEE:
            continue
        seen += 1
        front += int(((cx - cx[i]) > AHEAD * h[i]).sum()) == 0
    return seen, front


def call(pics, rule=(GUARD_T, GUARD_SEEN, GUARD_SHARE)):
    n, counts = vote(pics)
    info = {"candidate": n, "pictures_naming_each": counts}
    if n is None:
        return None, info
    seen, front = in_front(pics, n, rule[0])
    info.update(seen=seen, front=front)
    return (n if seen >= rule[1] and front >= rule[2] * seen else None), info


def rows(days, source=None):
    out = []
    for day in days:
        fc.FOLDER.setdefault(day, day)
        off = fc.official(day)
        for race, dr in sorted(dense(day, source).items()):
            rank = {h["number"]: h["rank"] for h in off[race]["horses"] if h.get("rank")}
            win = next((x for x, r in rank.items() if r == 1), None)
            n, info = call(dr["pictures"])
            n2, _ = call(dr["pictures"], SECOND)
            out.append({"day": day, "race": race, "clock_agrees": dr["clock_agrees"], "winner": win, "old": info["candidate"], "new": n, "second_setting": n2,
                        "rank_of_old": rank.get(info["candidate"]), "seen": info.get("seen"), "front": info.get("front"), "counts": info["pictures_naming_each"]})
    return out


def tally(rs, key):
    c = Counter("none" if r[key] is None else ("right" if r[key] == r["winner"] else "wrong") for r in rs)
    n = len(rs)
    called = c["right"] + c["wrong"]
    return {"races": n, "called": called, "right": c["right"], "wrong": c["wrong"], "none": c["none"],
            "called_share": round(100 * called / n, 1) if n else None, "right_of_calls": round(100 * c["right"] / called, 1) if called else None,
            "right_of_races": round(100 * c["right"] / n, 1) if n else None}


def main():
    mode = sys.argv[1]
    if mode == "same":
        day = sys.argv[2]
        a, b = rows([day], "winner"), rows([day], "caps2")
        diff = [(x["race"], x["old"], y["old"], x["new"], y["new"]) for x, y in zip(a, b) if (x["old"], x["new"], x["second_setting"]) != (y["old"], y["new"], y["second_setting"])]
        print(f"{day}: {len(a)} races compared, {len(diff)} with a different call", diff)
        return
    out, days = sys.argv[2], sys.argv[3:]
    rs = rows(days)
    if mode == "calls":
        Path(out).write_text(json.dumps({"rule": {"span": SPAN, "need": NEED, "guard_seconds": GUARD_T, "guard_seen": GUARD_SEEN, "guard_share": GUARD_SHARE},
                                         "days": {d: {r["race"]: {"call": r["new"], "candidate": r["old"], "seen": r["seen"], "front": r["front"]} for r in rs if r["day"] == d} for d in days}}, indent=1))
        print(len(rs), "races;", sum(r["new"] is not None for r in rs), "calls")
        return
    res = {"days": days, "rows": rs}
    for name, sel in (("all races run", rs), ("clock agrees", [r for r in rs if r["clock_agrees"]])):
        res[name] = {k: tally(sel, k) for k in ("old", "new", "second_setting")}
        print(name)
        for k, label in (("old", "call as v10 (no condition)"), ("new", "with the condition (primary)"), ("second_setting", "second setting (3 s, 2 pictures)")):
            t = res[name][k]
            print(f"  {label:34s} races {t['races']}: called {t['called']} ({t['called_share']}%), right {t['right']}, wrong {t['wrong']}  -> {t['right_of_calls']}% of calls, {t['right_of_races']}% of races")
    for d in days:
        t = {k: tally([r for r in rs if r["day"] == d], k) for k in ("old", "new")}
        print(f"  {d}: old {t['old']['right']} right {t['old']['wrong']} wrong of {t['old']['races']} | new {t['new']['right']} right {t['new']['wrong']} wrong")
    o, n = res["all races run"]["old"], res["all races run"]["new"]
    v1 = n["called"] >= 0.5 * n["races"] and n["called"] > 0 and n["right"] >= 0.9 * n["called"]
    v2 = n["wrong"] < o["wrong"] and n["right"] >= 0.8 * o["right"]
    res["verdicts"] = {"V1 the winner can be called (half of the races or more, 90% of calls or more right)": bool(v1),
                       "V2 the condition removes wrong calls (fewer wrong, at least 80% of the right calls kept)": bool(v2)}
    for k, v in res["verdicts"].items():
        print("VERDICT", k, "->", v)
    for r in rs:
        if r["old"] is not None and (r["old"] != r["winner"] or r["new"] is None):
            print("  ", r["day"], r["race"], "winner", r["winner"], "candidate", r["old"], f"(rank {r['rank_of_old']})", "seen/front", r["seen"], r["front"], "->", "dropped" if r["new"] is None else "kept")
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
