"""Overnight test on new race days: L1 again, the order rules again, and the fast configuration without Qwen. See night/DESIGN_KO.md.
usage: nt3_score.py OUT.json DAY_DIR [DAY_DIR ...]     reads da/DAY_DIR and the four model files; the reference is the broadcast strip.

Leading-group statements:  T1 (hand rule), L1 (current), F1 (no Qwen)
Leader "the leader is N":  HAND (lowest box), LEAD (tau 0.6), LEAD50 (tau 0.5), F_LEAD (F1, tau 0.6)
Order "a is ahead of b":   H0 (lower box, every pair), PAIR (margin 0.2), F_PAIR (F1, margin 0.2)
Addendum 2 (fallback where L1 says nothing): L1h = L1 + "hold", L1s = L1 + "hold_then_single" 0.8 (nf_common.py).
Addendum (more training days): L1x, F1x, LEADx, PAIRx = the same rules with models trained on six days instead of four."""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ld_common import leader_statements, pair_statements, pair_tally, pick_hand, pick_model, tally, yb_score  # noqa: E402
from nf_common import statements_fallback  # noqa: E402
from nq_common import ab, load_day_fast  # noqa: E402
from or_common import ROOT, load_day, statements  # noqa: E402

out_path, days = sys.argv[1], sys.argv[2:]
M = {k: joblib.load(ROOT / p) for k, p in (("l1", "order2/l1_model.joblib"), ("ld", "leader/leader_model.joblib"), ("l1f", "night/l1_fast.joblib"), ("ldf", "night/leader_fast.joblib"),
                                                ("l1x", "night/l1_six.joblib"), ("ldx", "night/leader_six.joblib"), ("l1fx", "night/l1_fast_six.joblib"))}
fn = {k: (lambda F, m=m: m.predict_proba(F)[:, 1]) for k, m in M.items()}
AB, LD, PR = ("T1", "L1", "F1", "L1x", "F1x", "L1h", "L1s"), ("HAND", "LEAD", "LEAD50", "F_LEAD", "LEADx"), ("H0", "PAIR", "F_PAIR", "PAIRx")


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


races, by_day = [], {}
for day in days:
    dl, df = load_day(f"da/{day}"), load_day_fast(f"da/{day}")
    said = {"T1": statements(dl, None), "L1": statements(dl, fn["l1"]), "F1": statements(df, fn["l1f"]),
            "L1x": statements(dl, fn["l1x"]), "F1x": statements(df, fn["l1fx"]),
            "L1h": statements_fallback(dl, fn["l1"], "hold")[0], "L1s": statements_fallback(dl, fn["l1"], "hold_then_single", 0.8)[0]}
    lead = {"HAND": leader_statements(dl, pick_hand, True), "LEAD": leader_statements(dl, lambda P: pick_model(P, fn["ld"], 0.6), True),
            "LEAD50": leader_statements(dl, lambda P: pick_model(P, fn["ld"], 0.5), True), "F_LEAD": leader_statements(df, lambda P: pick_model(P, fn["ldf"], 0.6), True),
            "LEADx": leader_statements(dl, lambda P: pick_model(P, fn["ldx"], 0.6), True)}
    pairs = {"H0": pair_statements(dl, said["L1"], yb_score, 0.0), "PAIR": pair_statements(dl, said["L1"], fn["ld"], 0.2), "F_PAIR": pair_statements(df, said["F1"], fn["ldf"], 0.2),
             "PAIRx": pair_statements(dl, said["L1x"], fn["ldx"], 0.2)}
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
                "statements": sum(u[k].get("statements", 0) for u in units), "points": sum(u[k].get("points", 0) for u in units)}
    for k in LD:
        s[k] = {"precision": ratio(units, k, "right", "statements"), "coverage": ratio(units, k, "right", "points"),
                "statements": sum(u[k].get("statements", 0) for u in units), "right": sum(u[k].get("right", 0) for u in units)}
    for k in PR:
        s[k] = {"right_share": ratio(units, k, "right", "scoreable"), "scoreable": sum(u[k].get("scoreable", 0) for u in units), "right": sum(u[k].get("right", 0) for u in units)}
    return s


pooled = summary(races)
days_sum = {d: summary([{k: v for k, v in r.items()} for r in by_day[d]["by_race"].values()]) for d in days}
iv = {"L1_A": boot(races, lambda u: ratio(u, "L1", "in_top3", "statements")), "L1_B": boot(races, lambda u: ratio(u, "L1", "good_points", "points")),
      "F1_minus_L1_A": boot(races, lambda u: diff(u, "F1", "L1", "in_top3", "statements")), "F1_minus_L1_B": boot(races, lambda u: diff(u, "F1", "L1", "good_points", "points")),
      "L1x_minus_L1_A": boot(races, lambda u: diff(u, "L1x", "L1", "in_top3", "statements")), "L1x_minus_L1_B": boot(races, lambda u: diff(u, "L1x", "L1", "good_points", "points")),
      "F1x_minus_F1_B": boot(races, lambda u: diff(u, "F1x", "F1", "good_points", "points")), "LEADx_minus_LEAD": boot(races, lambda u: diff(u, "LEADx", "LEAD", "right", "statements")),
      "L1h_minus_L1_A": boot(races, lambda u: diff(u, "L1h", "L1", "in_top3", "statements")), "L1h_minus_L1_B": boot(races, lambda u: diff(u, "L1h", "L1", "good_points", "points")),
      "L1s_minus_L1_A": boot(races, lambda u: diff(u, "L1s", "L1", "in_top3", "statements")), "L1s_minus_L1_B": boot(races, lambda u: diff(u, "L1s", "L1", "good_points", "points")),
      "L1_minus_T1_B": boot(races, lambda u: diff(u, "L1", "T1", "good_points", "points")),
      "LEAD": boot(races, lambda u: ratio(u, "LEAD", "right", "statements")), "LEAD50": boot(races, lambda u: ratio(u, "LEAD50", "right", "statements")),
      "LEAD_minus_HAND": boot(races, lambda u: diff(u, "LEAD", "HAND", "right", "statements")),
      "PAIR": boot(races, lambda u: ratio(u, "PAIR", "right", "scoreable")), "PAIR_minus_H0": boot(races, lambda u: diff(u, "PAIR", "H0", "right", "scoreable"))}


def each(cfg, key, bar):
    return all(days_sum[d][cfg][key] is not None and days_sum[d][cfg][key] >= bar for d in days)


P = pooled
verdict = {
    "L1_meets_goals_pooled": bool(P["L1"]["A"] >= 0.90 and P["L1"]["B"] >= 0.80),
    "L1_days_meeting_both": sum(days_sum[d]["L1"]["A"] >= 0.90 and days_sum[d]["L1"]["B"] >= 0.80 for d in days), "days": len(days),
    "leader_usable": bool(P["LEAD"]["precision"] >= 0.85 and each("LEAD", "precision", 0.80)),
    "leader50_usable": bool(P["LEAD50"]["precision"] >= 0.85 and each("LEAD50", "precision", 0.80)),
    "leader_more_accurate_than_hand": bool(iv["LEAD_minus_HAND"][0] > 0),
    "pair_usable": bool(P["PAIR"]["right_share"] >= 0.90 and each("PAIR", "right_share", 0.85)),
    "fast_meets_goals_pooled": bool(P["F1"]["A"] >= 0.90 and P["F1"]["B"] >= 0.80),
    "fast_not_worse_than_L1": bool(P["F1"]["A"] >= P["L1"]["A"] - 0.02 and P["F1"]["B"] >= P["L1"]["B"] - 0.03),
    "fast_leader_usable": bool(P["F_LEAD"]["precision"] >= 0.85 and each("F_LEAD", "precision", 0.80)),
    "more_days_help_L1": bool(iv["L1x_minus_L1_B"][0] > 0 and P["L1x"]["A"] >= P["L1"]["A"] - 0.01),
    "more_days_help_F1": bool(iv["F1x_minus_F1_B"][0] > 0 and P["F1x"]["A"] >= P["F1"]["A"] - 0.01),
    "L1x_meets_goals_pooled": bool(P["L1x"]["A"] >= 0.90 and P["L1x"]["B"] >= 0.80), "F1x_meets_goals_pooled": bool(P["F1x"]["A"] >= 0.90 and P["F1x"]["B"] >= 0.80),
    "leaderx_usable": bool(P["LEADx"]["precision"] >= 0.85 and each("LEADx", "precision", 0.80)),
    "hold_helps": bool(iv["L1h_minus_L1_B"][0] > 0 and P["L1h"]["A"] >= P["L1"]["A"] - 0.01 and P["L1h"]["A"] >= 0.90),
    "hold_then_single_helps": bool(iv["L1s_minus_L1_B"][0] > 0 and P["L1s"]["A"] >= P["L1"]["A"] - 0.01 and P["L1s"]["A"] >= 0.90),
    "fast_pair_usable": bool(P["F_PAIR"]["right_share"] >= 0.90 and each("F_PAIR", "right_share", 0.85)),
}
Path(out_path).write_text(json.dumps({"days": days, "races": len(races), "pooled": pooled, "by_day": days_sum, "intervals": iv, "verdict": verdict,
                                      "detail": by_day}, indent=1))
for d in days:
    s = days_sum[d]
    print(d, " ".join(f"{k} A {s[k]['A']:.3f} B {s[k]['B']:.3f} ({s[k]['points']} pts)" for k in AB), "|",
          " ".join(f"{k} {s[k]['right']}/{s[k]['statements']}" for k in LD), "|", " ".join(f"{k} {s[k]['right']}/{s[k]['scoreable']}" for k in PR))
print("pooled:", json.dumps(pooled))
print("intervals:", json.dumps(iv))
print("verdict:", json.dumps(verdict, indent=1))
