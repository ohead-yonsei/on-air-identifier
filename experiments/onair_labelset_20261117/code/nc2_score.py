"""Second confirmation of the night candidates: configuration F2 = no Qwen (F1) + fallback where nothing is said + leader threshold 0.5.
See night/confirm2/DESIGN_KO.md.   usage: nc2_score.py OUT.json DAY_DIR [DAY_DIR ...]     the reference is the broadcast strip.

Leading-group statements:  L1 (current), F1 (no Qwen), F2 (F1 + "hold_then_single" 0.8, nf_common.py), L1s (L1 + the same fallback)
Leader "the leader is N":  HAND, LEAD (0.6), LEAD50 (0.5), F_LEAD (F1 models, 0.6), F_LEAD50 (F1 models, 0.5)  <- F2 uses F_LEAD50
Order "a is ahead of b":   PAIR (L1 statements), F_PAIR (F1 statements), F2_PAIR (F2 statements), all margin 0.2"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ld_common import leader_statements, pair_statements, pair_tally, pick_hand, pick_model, tally  # noqa: E402
from nf_common import statements_fallback  # noqa: E402
from nq_common import ab, load_day_fast  # noqa: E402
from or_common import ROOT, load_day, statements  # noqa: E402

out_path, days = sys.argv[1], sys.argv[2:]
M = {k: joblib.load(ROOT / p) for k, p in (("l1", "order2/l1_model.joblib"), ("ld", "leader/leader_model.joblib"), ("l1f", "night/l1_fast.joblib"), ("ldf", "night/leader_fast.joblib"))}
fn = {k: (lambda F, m=m: m.predict_proba(F)[:, 1]) for k, m in M.items()}
AB, LD, PR = ("L1", "F1", "F2", "L1s"), ("HAND", "LEAD", "LEAD50", "F_LEAD", "F_LEAD50"), ("PAIR", "F_PAIR", "F2_PAIR")


def boot(units, f, n=2000, seed=20261118):
    rng, vals = random.Random(seed), []
    for _ in range(n):
        v = f([units[rng.randrange(len(units))] for _ in units])
        if v is not None:
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def ratio(units, cfg, num, den):
    a, b = sum(u[cfg].get(num, 0) for u in units), sum(u[cfg].get(den, 0) for u in units)
    return a / b if b else None


def diff(units, c1, c2, num, den):
    a, b = ratio(units, c1, num, den), ratio(units, c2, num, den)
    return None if a is None or b is None else a - b


races, by_day, added_total = [], {}, Counter()
for day in days:
    dl, df = load_day(f"da/{day}"), load_day_fast(f"da/{day}")
    f2, f2_added = statements_fallback(df, fn["l1f"], "hold_then_single", 0.8)
    said = {"L1": statements(dl, fn["l1"]), "F1": statements(df, fn["l1f"]), "F2": f2, "L1s": statements_fallback(dl, fn["l1"], "hold_then_single", 0.8)[0]}
    lead = {"HAND": leader_statements(dl, pick_hand, True), "LEAD": leader_statements(dl, lambda P: pick_model(P, fn["ld"], 0.6), True),
            "LEAD50": leader_statements(dl, lambda P: pick_model(P, fn["ld"], 0.5), True), "F_LEAD": leader_statements(df, lambda P: pick_model(P, fn["ldf"], 0.6), True),
            "F_LEAD50": leader_statements(df, lambda P: pick_model(P, fn["ldf"], 0.5), True)}
    pairs = {"PAIR": pair_statements(dl, said["L1"], fn["ld"], 0.2), "F_PAIR": pair_statements(df, said["F1"], fn["ldf"], 0.2), "F2_PAIR": pair_statements(df, said["F2"], fn["ldf"], 0.2)}
    for m, _ in dl:
        ex = f2_added[m["frame"]]
        added_total["added"] += len(ex)
        added_total["added_right"] += sum(n in m["strip"][:3] for n in ex)
    per = defaultdict(lambda: defaultdict(Counter))
    res = {}
    for k in AB:
        res[k], good = ab(dl, said[k])
        for m, _ in dl:
            r, nums = per[m["race"]][k], said[k][m["frame"]]
            r["points"] += 1
            r["statements"] += len(nums)
            r["in_top3"] += sum(n in m["strip"][:3] for n in nums)
            r["good_points"] += good[m["frame"]]
    for k in LD:
        res[k], good = tally(dl, lead[k])
        for m, _ in dl:
            r = per[m["race"]][k]
            r["points"] += 1
            r["statements"] += lead[k][m["frame"]] is not None
            r["right"] += good[m["frame"]]
    for k in PR:
        res[k], good = pair_tally(dl, pairs[k][0])
        res[k]["candidate_pairs"] = pairs[k][1]
        for m, _ in dl:
            r = per[m["race"]][k]
            r["scoreable"] += len(good[m["frame"]])
            r["right"] += sum(good[m["frame"]])
    by_day[day] = {"results": res, "by_race": {rc: {k: dict(v) for k, v in r.items()} for rc, r in sorted(per.items())}}
    races += [{k: dict(v) for k, v in r.items()} for r in per.values()]


def summary(units):
    s = {}
    for k in AB:
        s[k] = {"A": ratio(units, k, "in_top3", "statements"), "B": ratio(units, k, "good_points", "points"),
                "statements": sum(u[k].get("statements", 0) for u in units), "in_top3": sum(u[k].get("in_top3", 0) for u in units),
                "points": sum(u[k].get("points", 0) for u in units), "good_points": sum(u[k].get("good_points", 0) for u in units)}
    for k in LD:
        s[k] = {"precision": ratio(units, k, "right", "statements"), "coverage": ratio(units, k, "right", "points"),
                "statements": sum(u[k].get("statements", 0) for u in units), "right": sum(u[k].get("right", 0) for u in units), "points": sum(u[k].get("points", 0) for u in units)}
    for k in PR:
        s[k] = {"right_share": ratio(units, k, "right", "scoreable"), "scoreable": sum(u[k].get("scoreable", 0) for u in units), "right": sum(u[k].get("right", 0) for u in units)}
    return s


pooled = summary(races)
days_sum = {d: summary(list(by_day[d]["by_race"].values())) for d in days}
iv = {"F2_A": boot(races, lambda u: ratio(u, "F2", "in_top3", "statements")), "F2_B": boot(races, lambda u: ratio(u, "F2", "good_points", "points")),
      "F2_minus_F1_A": boot(races, lambda u: diff(u, "F2", "F1", "in_top3", "statements")), "F2_minus_F1_B": boot(races, lambda u: diff(u, "F2", "F1", "good_points", "points")),
      "F2_minus_L1_A": boot(races, lambda u: diff(u, "F2", "L1", "in_top3", "statements")), "F2_minus_L1_B": boot(races, lambda u: diff(u, "F2", "L1", "good_points", "points")),
      "F_LEAD50": boot(races, lambda u: ratio(u, "F_LEAD50", "right", "statements")), "F_LEAD50_minus_HAND": boot(races, lambda u: diff(u, "F_LEAD50", "HAND", "right", "statements")),
      "F2_PAIR": boot(races, lambda u: ratio(u, "F2_PAIR", "right", "scoreable"))}


def each(cfg, key, bar):
    return all(days_sum[d][cfg][key] is not None and days_sum[d][cfg][key] >= bar for d in days)


P = pooled
verdict = {
    "F2_meets_goals_pooled": bool(P["F2"]["A"] >= 0.90 and P["F2"]["B"] >= 0.80),
    "F2_days_meeting_both": sum(days_sum[d]["F2"]["A"] >= 0.90 and days_sum[d]["F2"]["B"] >= 0.80 for d in days), "days": len(days),
    "F2_not_worse_than_L1": bool(P["F2"]["A"] >= P["L1"]["A"] - 0.02 and P["F2"]["B"] >= P["L1"]["B"] - 0.03),
    "fallback_helps_fast": bool(iv["F2_minus_F1_B"][0] > 0 and P["F2"]["A"] >= P["F1"]["A"] - 0.01 and P["F2"]["A"] >= 0.90),
    "fast_leader50_usable": bool(P["F_LEAD50"]["precision"] >= 0.85 and each("F_LEAD50", "precision", 0.80)),
    "F2_pair_usable": bool(P["F2_PAIR"]["right_share"] >= 0.90 and each("F2_PAIR", "right_share", 0.85)),
}
verdict["F2_recommended_as_default"] = bool(verdict["F2_meets_goals_pooled"] and verdict["F2_days_meeting_both"] == len(days) and verdict["F2_not_worse_than_L1"]
                                            and verdict["fast_leader50_usable"] and verdict["F2_pair_usable"])
Path(out_path).write_text(json.dumps({"days": days, "races": len(races), "pooled": pooled, "by_day": days_sum, "intervals": iv, "verdict": verdict,
                                      "fallback_added_on_F2": dict(added_total), "detail": by_day}, indent=1))
for d in days:
    s = days_sum[d]
    print(d, " ".join(f"{k} A {s[k]['A']:.3f} B {s[k]['B']:.3f}" for k in AB), f"({s['L1']['points']} pts) |",
          " ".join(f"{k} {s[k]['right']}/{s[k]['statements']}" for k in LD), "|", " ".join(f"{k} {s[k]['right']}/{s[k]['scoreable']}" for k in PR))
print("pooled:", " ".join(f"{k} A {P[k]['in_top3']}/{P[k]['statements']}={P[k]['A']:.3f} B {P[k]['good_points']}/{P[k]['points']}={P[k]['B']:.3f}" for k in AB))
print("       ", " ".join(f"{k} {P[k]['right']}/{P[k]['statements']}={P[k]['precision']:.3f} cov {P[k]['coverage']:.3f}" for k in LD))
print("       ", " ".join(f"{k} {P[k]['right']}/{P[k]['scoreable']}={P[k]['right_share']:.3f}" for k in PR))
print("fallback added on F2:", dict(added_total))
print("intervals:", json.dumps({k: [round(x, 4) for x in v] for k, v in iv.items()}))
print("verdict:", json.dumps(verdict, indent=1))
