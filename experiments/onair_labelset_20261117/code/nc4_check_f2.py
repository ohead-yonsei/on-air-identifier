"""Check that identifier/identify_l1.py --fast --fallback --leader-tau 0.5 (configuration F2) gives the same statements, leader and order
as the scoring script's F2 (nc2_score.py: nq_common + nf_common on da/DAY).
usage: nc4_check_f2.py DAY_DIR OUT.json     OUT.json = output of identify_l1.py --day-dir DAY_DIR --fast --fallback --leader-tau 0.5"""
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ld_common import leader_statements, pair_statements, pick_model  # noqa: E402
from nf_common import statements_fallback  # noqa: E402
from nq_common import load_day_fast  # noqa: E402
from or_common import ROOT  # noqa: E402

day, out = sys.argv[1], json.loads(Path(sys.argv[2]).read_text())
assert out["configuration"] == "F2", out["configuration"]
l1 = joblib.load(ROOT / "night" / "l1_fast.joblib")
ld = joblib.load(ROOT / "night" / "leader_fast.joblib")
ldfn = lambda F: ld.predict_proba(F)[:, 1]  # noqa: E731
data = load_day_fast(f"da/{day}")
stated, added = statements_fallback(data, lambda F: l1.predict_proba(F)[:, 1], "hold_then_single", 0.8)
lead = leader_statements(data, lambda P: pick_model(P, ldfn, 0.5), True)
pairs, _ = pair_statements(data, stated, ldfn, 0.2)
n = same_s = same_l = same_o = fb = same_fb = 0
for p in out["points"]:
    f = p["frame"]
    n += 1
    a, b = sorted(pairs[f]), sorted((o["ahead"], o["behind"]) for o in p["order"])
    same_s += sorted(stated[f]) == p["statements"]
    same_l += lead[f] == p["leader"]
    same_o += a == b
    mine = sorted(d["number"] for d in p["detail"] if d["source"] == "fallback")
    fb += bool(added[f])
    same_fb += mine == sorted(added[f])
print(f"{day}: time points {n}; identical statements {same_s}, identical leader {same_l}, identical order {same_o}; "
      f"points with a fallback number {fb}, identical fallback numbers {same_fb}; program seconds {out.get('seconds')}")
