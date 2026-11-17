"""Check that identifier/identify_l1.py --fast gives the same statements, leader and order as the scoring scripts' F1 (nq_common on da/DAY).
usage: nt4_check_fast.py DAY_DIR OUT.json     OUT.json = output of identify_l1.py --day-dir DAY_DIR --fast"""
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ld_common import leader_statements, pair_statements, pick_model  # noqa: E402
from nq_common import load_day_fast  # noqa: E402
from or_common import ROOT, statements  # noqa: E402

day, out = sys.argv[1], json.loads(Path(sys.argv[2]).read_text())
l1 = joblib.load(ROOT / "night" / "l1_fast.joblib")
ld = joblib.load(ROOT / "night" / "leader_fast.joblib")
ldfn = lambda F: ld.predict_proba(F)[:, 1]  # noqa: E731
data = load_day_fast(f"da/{day}")
stated = statements(data, lambda F: l1.predict_proba(F)[:, 1])
lead = leader_statements(data, lambda P: pick_model(P, ldfn, 0.6), True)
pairs, _ = pair_statements(data, stated, ldfn, 0.2)
n = same_s = same_l = same_o = 0
for p in out["points"]:
    f = p["frame"]
    n += 1
    a, b = sorted(pairs[f]), sorted((o["ahead"], o["behind"]) for o in p["order"])
    same_s += sorted(stated[f]) == p["statements"]
    same_l += lead[f] == p["leader"]
    same_o += a == b
print(f"{day}: time points {n}; identical statements {same_s}, identical leader {same_l}, identical order {same_o}; program seconds {out.get('seconds')}")
