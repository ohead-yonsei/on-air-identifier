"""Test of the learned order rules on days the leader model never saw. See leader/DESIGN_KO.md.
usage: ld3_score.py DAY_DIR [DAY_DIR ...]     reads da/DAY_DIR, order2/l1_model.joblib, leader/leader_model.joblib; writes leader/score.json

Part 1 "the leader is N":   LEAD = leader model, tau 0.6, carry      vs  HAND = lowest box if sure, carry
Part 2 "a is ahead of b":   PAIR = leader-model score, margin 0.2    vs  H0 = lower box, every pair;  H2 = lower box, margin 0.02
The reference is the broadcast strip (first five). Pairs are taken among the numbers the identifier (L1) states at that time point."""
import json
import random
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ld_common import leader_statements, pair_statements, pair_tally, pick_hand, pick_model, tally, yb_score  # noqa: E402
from or_common import ROOT, load_day, statements  # noqa: E402

TAU_LEAD, MARGIN_PAIR, MARGIN_H2 = 0.6, 0.2, 0.02
days = sys.argv[1:]
l1 = joblib.load(ROOT / "order2" / "l1_model.joblib")
ld = joblib.load(ROOT / "leader" / "leader_model.joblib")
l1fn = lambda F: l1.predict_proba(F)[:, 1]  # noqa: E731
ldfn = lambda F: ld.predict_proba(F)[:, 1]  # noqa: E731


def sign_p(a, b):
    n, mm = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, k) for k in range(mm + 1)) / 2 ** n) if n else 1.0


def boot(units, f, n=2000, seed=20261118):
    """95% interval of f over races resampled with replacement. units = list of per-race records."""
    rng, vals = random.Random(seed), []
    for _ in range(n):
        v = f([units[rng.randrange(len(units))] for _ in units])
        if v is not None:
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def add(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


by_day, races = {}, []          # races: one record per (day, race) for the bootstrap
pooled = defaultdict(dict)
wrong_place = Counter()
ref = defaultdict(lambda: defaultdict(dict))
flips = Counter()
for day in days:
    data = load_day(f"da/{day}")
    stated = statements(data, l1fn)
    L = {"LEAD": leader_statements(data, lambda P: pick_model(P, ldfn, TAU_LEAD), True), "HAND": leader_statements(data, pick_hand, True)}
    Pp = {"PAIR": pair_statements(data, stated, ldfn, MARGIN_PAIR), "H0": pair_statements(data, stated, yb_score, 0.0), "H2": pair_statements(data, stated, yb_score, MARGIN_H2)}
    res, lgood, pgood = {}, {}, {}
    for k, said in L.items():
        res[k], lgood[k] = tally(data, said)
    for k, (pairs, cand) in Pp.items():
        res[k], pgood[k] = pair_tally(data, pairs)
        res[k]["candidate_pairs"] = cand
        res[k]["points_with_pair"] = sum(bool(v) for v in pairs.values())
        res[k]["points_with_wrong_pair"] = sum(not all(v) for v in pgood[k].values())
    flips["lead_gain"] += sum(lgood["LEAD"][f] and not lgood["HAND"][f] for f in lgood["LEAD"])
    flips["lead_loss"] += sum(lgood["HAND"][f] and not lgood["LEAD"][f] for f in lgood["LEAD"])
    for m, _ in data:                                    # where the wrong leader statements actually were on the strip
        s = L["LEAD"][m["frame"]]
        if s is not None and s != m["strip"][0]:
            wrong_place["second" if s == m["strip"][1] else "third" if s == m["strip"][2] else "fourth_or_fifth" if s in m["strip"] else "not_on_strip"] += 1
    per_race = defaultdict(lambda: defaultdict(Counter))
    for m, _ in data:
        f, r = m["frame"], per_race[m["race"]]
        for k in L:
            s = L[k][f]
            r[k]["points"] += 1
            r[k]["statements"] += s is not None
            r[k]["right"] += lgood[k][f]
        for k in Pp:
            r[k]["scoreable"] += len(pgood[k][f])
            r[k]["right"] += sum(pgood[k][f])
            r[k]["said"] += len(Pp[k][0][f])
    races += [{k: dict(v) for k, v in r.items()} for r in per_race.values()]
    by_day[day] = {"results": res, "by_race": {rc: {k: dict(v) for k, v in r.items()} for rc, r in sorted(per_race.items())}}
    for k in res:
        pooled[k] = add(pooled[k], res[k])
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7):                # REFERENCE ONLY (not used for any verdict): other thresholds on the test days
        ref["leader_tau"][tau] = add(ref["leader_tau"][tau], tally(data, leader_statements(data, lambda P, tau=tau: pick_model(P, ldfn, tau), True))[0])
    for mg in (0.0, 0.1, 0.2, 0.3, 0.4):
        pairs, cand = pair_statements(data, stated, ldfn, mg)
        ref["pair_margin"][mg] = add(ref["pair_margin"][mg], {**pair_tally(data, pairs)[0], "candidate_pairs": cand})


def prec(r):
    return r["right"] / r["statements"] if r["statements"] else None


def pprec(r):
    return r["right"] / r["scoreable"] if r["scoreable"] else None


def agg(units, key, den):
    a, b = sum(u[key].get("right", 0) for u in units), sum(u[key].get(den, 0) for u in units)
    return a / b if b else None


def diff(units, k1, k2, den):
    a, b = agg(units, k1, den), agg(units, k2, den)
    return None if a is None or b is None else a - b


P = pooled
lead_days = [prec(by_day[d]["results"]["LEAD"]) for d in days]
pair_days = [pprec(by_day[d]["results"]["PAIR"]) for d in days]
ci_lead = boot(races, lambda u: diff(u, "LEAD", "HAND", "statements"))
ci_pair0 = boot(races, lambda u: diff(u, "PAIR", "H0", "scoreable"))
ci_pair2 = boot(races, lambda u: diff(u, "PAIR", "H2", "scoreable"))
verdict = {
    "leader_usable": bool(prec(P["LEAD"]) >= 0.85 and all(v is not None and v >= 0.80 for v in lead_days)),
    "leader_precision": prec(P["LEAD"]), "leader_precision_by_day": lead_days, "leader_coverage": P["LEAD"]["right"] / P["LEAD"]["points"],
    "leader_precision_interval": boot(races, lambda u: agg(u, "LEAD", "statements")),
    "hand_precision": prec(P["HAND"]), "hand_coverage": P["HAND"]["right"] / P["HAND"]["points"],
    "leader_more_accurate_than_hand": bool(ci_lead[0] > 0), "leader_minus_hand_precision": prec(P["LEAD"]) - prec(P["HAND"]), "leader_minus_hand_interval": ci_lead,
    "leader_points_gained": flips["lead_gain"], "leader_points_lost": flips["lead_loss"], "leader_points_sign_test_p": sign_p(flips["lead_gain"], flips["lead_loss"]),
    "wrong_leader_statements_were": dict(wrong_place),
    "pair_usable": bool(pprec(P["PAIR"]) >= 0.90 and all(v is not None and v >= 0.85 for v in pair_days)),
    "pair_right": pprec(P["PAIR"]), "pair_right_by_day": pair_days, "pair_share_of_candidates_said": P["PAIR"]["said"] / max(1, P["PAIR"]["candidate_pairs"]),
    "pair_right_interval": boot(races, lambda u: agg(u, "PAIR", "scoreable")),
    "hand_all_pairs_right": pprec(P["H0"]), "hand_margin_right": pprec(P["H2"]), "hand_margin_share_said": P["H2"]["said"] / max(1, P["H2"]["candidate_pairs"]),
    "pair_better_than_hand": bool(ci_pair0[0] > 0 and pprec(P["PAIR"]) > pprec(P["H2"]) and P["PAIR"]["said"] >= P["H2"]["said"]),
    "pair_minus_hand_all_interval": ci_pair0, "pair_minus_hand_margin_interval": ci_pair2,
}
out = {"days": days, "races": len(races), "pooled": {k: dict(v) for k, v in P.items()}, "by_day": by_day, "verdict": verdict,
       "reference_only": {a: {str(k): v for k, v in b.items()} for a, b in ref.items()}}
(ROOT / "leader" / "score.json").write_text(json.dumps(out, indent=1))
for d in days:
    r = by_day[d]["results"]
    print(d, "| leader:", "  ".join(f"{k} {r[k]['right']}/{r[k]['statements']} of {r[k]['points']} points" for k in ("LEAD", "HAND")),
          "| pairs:", "  ".join(f"{k} {r[k]['right']}/{r[k]['scoreable']} (said {r[k]['said']} of {r[k]['candidate_pairs']})" for k in ("PAIR", "H0", "H2")))
print("pooled:", json.dumps({k: dict(v) for k, v in P.items()}))
print("verdict:", json.dumps(verdict, indent=1))
print("reference only:", json.dumps(out["reference_only"]))
