"""L1 with a fallback for time points where it says nothing (configuration L1s). See night/DESIGN_KO.md (addendum 2).
The usual statements are made first (or_common.statements). Only if that set is empty, one number may be added by a fallback rule:
  "hold"   the number(s) L1 stated 3 s earlier from its own pictures, if that number is sure in at least one of the 3 current pictures
  "single" the sure rider with the highest front-of-field probability over the 3 current pictures, if that probability is at least t1
  "hold_then_single" hold; if hold adds nothing, single"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from or_common import K, TAU, THR  # noqa: E402


def statements_fallback(day_data, score_fn, rule, t1=0.8):
    base, best, sure_now = {}, {}, {}
    for m, pics in day_data:
        votes, top, seen = Counter(), {}, set()
        for L in pics.values():
            sure = [r for r in L if r[1] >= THR]
            if not sure:
                continue
            pr = score_fn(np.array([r[3] for r in sure]))
            for r, v in zip(sure, pr):
                seen.add(r[0])
                top[r[0]] = max(top.get(r[0], 0.0), float(v))
            for i in np.argsort(-pr)[:K]:
                if pr[i] >= TAU:
                    votes[sure[i][0]] += 1
        base[m["frame"]] = {n for n, v in votes.items() if v >= 2}
        best[m["frame"]], sure_now[m["frame"]] = top, seen
    out, added = {}, {}
    for m, _ in day_data:
        f = m["frame"]
        race, t = f.split("_t")
        prev = [base.get(f"{race}_t{int(t) - b:03d}", set()) for b in (3, 6)]
        s = base[f] | (prev[0] & prev[1])
        extra = set()
        if not s:
            if rule in ("hold", "hold_then_single"):
                extra = prev[0] & sure_now[f]
            if not extra and rule in ("single", "hold_then_single") and best[f]:
                n = max(best[f], key=best[f].get)
                if best[f][n] >= t1:
                    extra = {n}
        out[f], added[f] = s | extra, extra
    return out, added
