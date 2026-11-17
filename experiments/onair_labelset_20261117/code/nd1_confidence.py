"""How far can each statement be trusted? A description (made after the night results were seen) of the identifier's own signals against the strip.
usage: nd1_confidence.py OUT.json DAY_DIR [DAY_DIR ...]     meant for days the models were not trained on (otherday8-11).
For L1 (Qwen) and F1 (no Qwen): leading-group statements by where they came from and by the model's probability; leader statements by score.
Nothing is tuned here; the thresholds are the ones already in use."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nq_common import load_day_fast  # noqa: E402
from or_common import K, ROOT, TAU, THR, load_day  # noqa: E402

out_path, days = sys.argv[1], sys.argv[2:]
CFG = {"L1": (load_day, "order2/l1_model.joblib", "leader/leader_model.joblib"), "F1": (load_day_fast, "night/l1_fast.joblib", "night/leader_fast.joblib")}


def pbin(v, lo):
    edges = [lo, 0.7, 0.8, 0.9, 0.95]
    edges = sorted(set(e for e in edges if e >= lo))
    for a, b in zip(edges, edges[1:] + [1.01]):
        if a <= v < b:
            return f"{a:.2f}-{min(b, 1.0):.2f}"
    return "below"


res = {}
for name, (loader, m1, m2) in CFG.items():
    l1, ld = joblib.load(ROOT / m1), joblib.load(ROOT / m2)
    S, P, LS, NP = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for day in days:
        data = loader(f"da/{day}")
        base, top, sure_now, lead_pick, lead_top = {}, {}, {}, {}, {}
        for m, pics in data:
            votes, best, seen, lv, lt = Counter(), {}, set(), Counter(), {}
            for L in pics.values():
                sure = [r for r in L if r[1] >= THR]
                if not sure:
                    continue
                F = np.array([r[3] for r in sure])
                pr, pf = l1.predict_proba(F)[:, 1], ld.predict_proba(F)[:, 1]
                for r, v in zip(sure, pr):
                    seen.add(r[0])
                    best[r[0]] = max(best.get(r[0], 0.0), float(v))
                for i in np.argsort(-pr)[:K]:
                    if pr[i] >= TAU:
                        votes[sure[i][0]] += 1
                i = int(np.argmax(pf))
                if pf[i] >= 0.5:
                    lv[sure[i][0]] += 1
                    lt[sure[i][0]] = max(lt.get(sure[i][0], 0.0), float(pf[i]))
            f = m["frame"]
            base[f], top[f], sure_now[f] = {n: v for n, v in votes.items() if v >= 2}, best, seen
            pk = [n for n, v in lv.items() if v >= 2]
            lead_pick[f], lead_top[f] = (pk[0] if pk else None), lt
        for m, _ in data:
            f, B = m["frame"], m["strip"]
            race, t = f.split("_t")
            prev = [set(base.get(f"{race}_t{int(t) - b:03d}", {})) for b in (3, 6)]
            now = base[f]
            carried = (prev[0] & prev[1]) - set(now)
            for n, v in now.items():
                ok = n in B[:3]
                S[f"now, picked in {v} of 3 pictures"]["n"] += 1
                S[f"now, picked in {v} of 3 pictures"]["right"] += ok
                P[pbin(top[f][n], TAU)]["n"] += 1
                P[pbin(top[f][n], TAU)]["right"] += ok
            for n in carried:
                S["carried from 3 s and 6 s earlier"]["n"] += 1
                S["carried from 3 s and 6 s earlier"]["right"] += n in B[:3]
            said = set(now) | carried
            if not said:
                extra, kind = prev[0] & sure_now[f], "fallback: held from 3 s earlier"
                if not extra and top[f]:
                    n = max(top[f], key=top[f].get)
                    if top[f][n] >= 0.8:
                        extra, kind = {n}, "fallback: single rider at 0.8 or more"
                for n in extra:
                    S[kind]["n"] += 1
                    S[kind]["right"] += n in B[:3]
            k = "3 or more" if len(said) >= 3 else str(len(said))
            NP[k]["n"] += 1
            NP[k]["good"] += any(n in B[:3] for n in said)
            ld_now = lead_pick[f]
            if ld_now is not None:
                key = "now, top score " + pbin(lead_top[f][ld_now], 0.5)
                LS[key]["n"] += 1
                LS[key]["right"] += ld_now == B[0]
                LS[key]["in_first_two"] += ld_now in B[:2]
            else:
                a, b = (lead_pick.get(f"{race}_t{int(t) - back:03d}") for back in (3, 6))
                if a is not None and a == b:
                    LS["carried from 3 s and 6 s earlier"]["n"] += 1
                    LS["carried from 3 s and 6 s earlier"]["right"] += a == B[0]
                    LS["carried from 3 s and 6 s earlier"]["in_first_two"] += a in B[:2]
    res[name] = {"statements_by_source": {k: dict(v) for k, v in sorted(S.items())}, "statements_now_by_probability": {k: dict(v) for k, v in sorted(P.items())},
                 "points_by_number_of_statements": {k: dict(v) for k, v in sorted(NP.items())}, "leader_tau_0.5_by_score": {k: dict(v) for k, v in sorted(LS.items())}}
    print("==", name)
    for title, tab, key in (("statements by source", S, "right"), ("statements made now, by highest leading-three probability", P, "right"),
                            ("time points by number of statements (good = at least one right)", NP, "good"), ("leader statements (threshold 0.5) by score", LS, "right")):
        print(" ", title)
        for k, v in sorted(tab.items()):
            extra = f"  first two {v['in_first_two']}" if "in_first_two" in v else ""
            print(f"    {k:42s} {v[key]:4d} / {v['n']:4d} = {v[key] / v['n']:.3f}{extra}")
Path(out_path).write_text(json.dumps({"days": days, "note": "description made after the results were seen; nothing tuned", "configurations": res}, indent=1))
