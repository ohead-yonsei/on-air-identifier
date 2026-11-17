"""POST-HOC description (not a test): what the wrong and the missing statements of L1 look like. usage: nt6_errors.py OUT.json DAY_DIR [...]
For every strip time point of the given days (da/DAY_DIR):
  wrong statement = a stated number that is not in the strip's first three; split into "fourth or fifth on the strip" and "not on the strip"
  missed point    = a time point with no right statement; split into "said nothing" and "said only wrong numbers"
and how these go with the number (cap colour), the number of riders found, and the size of the riders in the picture."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from or_common import ROOT, THR, load_day, statements  # noqa: E402

out_path, days = sys.argv[1], sys.argv[2:]
l1 = joblib.load(ROOT / "order2" / "l1_model.joblib")
by_num = defaultdict(Counter)
kinds, miss = Counter(), Counter()
bins = defaultdict(Counter)        # by median rider height in the newest picture
nrid = defaultdict(Counter)        # by number of riders in the newest picture
first_num = defaultdict(Counter)   # by the strip's first number: is the point good?
per_day = {}
for day in days:
    data = load_day(f"da/{day}")
    said = statements(data, lambda F: l1.predict_proba(F)[:, 1])
    d = Counter()
    for m, pics in data:
        S, nums = m["strip"], said[m["frame"]]
        L = pics["t"]
        hs = [r[3][10] * 1080 for r in L]
        hb = "no riders" if not hs else "under 60 px" if np.median(hs) < 60 else "60-100 px" if np.median(hs) < 100 else "100-160 px" if np.median(hs) < 160 else "160 px or more"
        nb = "0-3" if len(L) < 4 else "4-6" if len(L) <= 6 else "7-9" if len(L) <= 9 else "10 or more"
        good = any(n in S[:3] for n in nums)
        d["points"] += 1
        d["good"] += good
        for key, table in ((hb, bins), (nb, nrid)):
            table[key]["points"] += 1
            table[key]["good"] += good
            table[key]["statements"] += len(nums)
            table[key]["right"] += sum(n in S[:3] for n in nums)
        first_num[S[0]]["points"] += 1
        first_num[S[0]]["good"] += good
        first_num[S[0]]["first_stated"] += S[0] in nums
        sure_now = {r[0] for LL in pics.values() for r in LL if r[1] >= THR}
        first_num[S[0]]["first_sure_in_a_picture"] += S[0] in sure_now
        if not good:
            miss["said nothing" if not nums else "said only wrong numbers"] += 1
        for n in nums:
            by_num[n]["stated"] += 1
            if n in S[:3]:
                by_num[n]["right"] += 1
            else:
                k = "fourth or fifth on the strip" if n in S else "not on the strip"
                kinds[k] += 1
                by_num[n][k] += 1
    per_day[day] = dict(d)
res = {"days": days, "per_day": per_day, "wrong_statements": dict(kinds), "missed_points": dict(miss),
       "by_stated_number": {str(k): dict(v) for k, v in sorted(by_num.items())}, "by_rider_height": {k: dict(v) for k, v in bins.items()},
       "by_riders_found": {k: dict(v) for k, v in nrid.items()}, "by_true_leader_number": {str(k): dict(v) for k, v in sorted(first_num.items())}}
Path(out_path).write_text(json.dumps(res, indent=1))
print("wrong statements:", dict(kinds), "| missed points:", dict(miss))
print("by stated number (stated, right, 4th-5th, off strip):")
for k, v in sorted(by_num.items()):
    print(f"  {k:2d}: {v['stated']:4d} {v['right']:4d} {v['fourth or fifth on the strip']:3d} {v['not on the strip']:3d}  right {v['right'] / v['stated']:.2f}")
print("by median rider height:", {k: f"A {v['right']}/{v['statements']} B {v['good']}/{v['points']}" for k, v in bins.items()})
print("by riders found:", {k: f"A {v['right']}/{v['statements']} B {v['good']}/{v['points']}" for k, v in nrid.items()})
print("by the strip's first number (points, good, first stated, first sure in a picture):")
for k, v in sorted(first_num.items()):
    print(f"  {k:2d}: {v['points']:4d} good {v['good'] / v['points']:.2f}  leader stated {v['first_stated'] / v['points']:.2f}  leader's number sure in some picture {v['first_sure_in_a_picture'] / v['points']:.2f}")
