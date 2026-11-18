"""finish3 stage 2, second exploration (days seen before, after g5_explore.py found the whole-race models worse than the base).
usage: g8_explore2.py OUT.json            (CPU only)

1. Why is no leader named? For every time point: is the true leader's number readable at all, i.e. is it a sure rider (cap probability 0.7
   or more) in at least 2 of the 3 pictures? The share of such points is the ceiling for "leader right share" with these pictures.
   Then, among readable points: named right / named wrong / silent; among unreadable points: named wrong / silent.
2. Cheap variants of the base (v10), all with the existing models:
   thr T          v10 with threshold T for both models (base: 0.5)
   avg T          the scores of the base's model and of the whole-race model W19 are averaged, threshold T
   union T        v10's leader; where v10 is silent, the whole-race model W19's own pick of that time point at threshold T
   one of three   v10, but a number named in only 1 of the 3 pictures also counts when no other number is named in them
Sets of days and the leave-day-out handling as in g5_explore.py. Everything here is exploration on seen days."""
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import g5_explore as g5

PARTS = g5.PARTS


def pick_from(e, s, thr):
    if e is None or not e["X"]:
        return None
    j = int(np.argmax(s))
    return e["nums"][j] if s[j] >= thr else None


def race_rows(rr, variant):
    tw, pics, rows = rr["tw"], rr["pics"], []
    taus = list(range(3 * int((tw - 9.0) // 3), -1, -3))
    kind, thr = variant

    def three(tau):
        return [pics.get(round(tau + o, 1)) for o in (0.0, 0.5, 1.0)]

    def now(tau, score_of, t, loose=False):
        ps = [pick_from(e, score_of(e), t) if e is not None and e["X"] else None for e in three(tau)]
        c = Counter(n for n in ps if n is not None)
        n = next((n for n, v in c.most_common() if v >= 2), None)
        if n is None and loose and len(c) == 1:
            n = next(iter(c))
        return n

    def model_at(tau, extra=None):
        name = "fs" if tau <= 36 else "strip"
        if extra is None:
            return lambda e: e["s"][name]
        return lambda e: (e["s"][name] + e["s"][extra]) / 2

    own, own_late, own_w = {}, {}, {}
    for tau in taus:
        t = thr if kind in ("thr", "avg") else 0.5
        extra = "W19" if kind == "avg" else None
        loose = kind == "one of three"
        own[tau] = now(tau, (lambda e: e["s"]["strip"]) if extra is None else (lambda e: (e["s"]["strip"] + e["s"]["W19"]) / 2), t, loose)
        lead = own[tau]
        if lead is None and own.get(tau + 3) is not None and own.get(tau + 3) == own.get(tau + 6):
            lead = own[tau + 3]
        if tau <= 36:
            own_late[tau] = now(tau, model_at(tau, extra), t, loose)
            lead = own_late[tau]
            if lead is None and own_late.get(tau + 3) is not None and own_late.get(tau + 3) == own_late.get(tau + 6):
                lead = own_late[tau + 3]
        if kind == "union" and lead is None:
            lead = now(tau, lambda e: e["s"]["W19"], thr)
        e0 = pics.get(float(tau))
        ref = e0["ref"] if e0 else None
        if ref is None:
            continue
        readable = sum(1 for e in three(tau) if e is not None and ref["leader"] in e["nums"]) >= 2
        rt = tw - tau
        rows.append({"tau": tau, "part": "start" if rt < 18 else ("finish" if tau <= 36 else "middle"), "said": lead is not None,
                     "right": lead is not None and lead == ref["leader"], "lenient": lead is not None and lead in ref["leader_lenient"], "readable": readable,
                     "sure_riders": int(np.mean([len(e["nums"]) if e else 0 for e in three(tau)]) + 0.5)})
    return rows


def main():
    out = Path(sys.argv[1])
    data = {d: g5.load(d) for d in g5.DEV + g5.HELD}
    strip = joblib.load(ROOT / "night" / "leader_fast.joblib")
    for d in g5.DEV:
        X, y = g5.samples(data, [x for x in g5.DEV if x != d])
        g5.score_day(data[d], {"strip": (strip, 19), "fs": (joblib.load(ROOT / "finish2" / "models" / f"fs_leader_without_{d}.joblib"), 19), "W19": (g5.fit(X[:, :19], y), 19)})
        print("scored", d, flush=True)
    w19, fs_all = joblib.load(ROOT / "finish3" / "models" / "W19_all.joblib"), joblib.load(ROOT / "finish2" / "models" / "fs_leader_all.joblib")
    for d in g5.HELD:
        g5.score_day(data[d], {"strip": (strip, 19), "fs": (fs_all, 19), "W19": (w19, 19)})
    variants = [("thr", 0.5)] + [("thr", t) for t in (0.4, 0.3, 0.2)] + [("avg", t) for t in (0.5, 0.4, 0.3)] + [("union", t) for t in (0.5, 0.4, 0.3)] + [("one of three", 0.5)]
    res = {}
    for key, days in (("dev", g5.DEV), ("held", g5.HELD)):
        res[key] = {}
        runs = {f"{k} {t}": {(d, race): race_rows(rr, (k, t)) for d in days for race, rr in data[d].items() if rr["agree"]} for k, t in variants}
        base = runs["thr 0.5"]
        rows = [r for v in base.values() for r in v]
        print(f"\n== {key}: {len(base)} races, {len(rows)} time points")
        diag = {}
        for p in PARTS:
            rs = rows if p == "whole race" else [r for r in rows if r["part"] == p]
            rd, un = [r for r in rs if r["readable"]], [r for r in rs if not r["readable"]]
            diag[p] = {"points": len(rs), "readable": len(rd), "readable_right": sum(r["right"] for r in rd), "readable_wrong": sum(r["said"] and not r["right"] for r in rd),
                       "readable_silent": sum(not r["said"] for r in rd), "unreadable_right": sum(r["right"] for r in un),
                       "unreadable_wrong": sum(r["said"] and not r["right"] for r in un), "unreadable_silent": sum(not r["said"] for r in un)}
            v = diag[p]
            print(f"  {p:11s} points {v['points']:4d}: true leader readable in {v['readable']} ({100 * v['readable'] / max(v['points'], 1):.1f}%) -> right {v['readable_right']}, wrong {v['readable_wrong']}, silent {v['readable_silent']}"
                  f" | not readable {v['points'] - v['readable']} -> right (carried) {v['unreadable_right']}, wrong {v['unreadable_wrong']}, silent {v['unreadable_silent']}")
        res[key]["why_no_leader (base)"] = diag
        for name, by in runs.items():
            t = g5.table(by)
            w = t["whole race"]
            extra = {} if name == "thr 0.5" else {"right_share_minus_base": g5.paired(by, base), "wrong_share_minus_base": g5.paired(by, base, "wrong_share")}
            res[key][name] = {"parts": t, **extra}
            print(f"{name:18s} whole race: right share {w['right_share']}%  wrong share {w['wrong_share']}%  when said {w['when_said']}% ({w['right']}/{w['said']})"
                  f" | start {t['start']['right_share']}/{t['start']['wrong_share']}  middle {t['middle']['right_share']}/{t['middle']['wrong_share']}  finish {t['finish']['right_share']}/{t['finish']['wrong_share']}"
                  + ("" if not extra else f" | minus base: right {extra['right_share_minus_base']} wrong {extra['wrong_share_minus_base']}"))
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
