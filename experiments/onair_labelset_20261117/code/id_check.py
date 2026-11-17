"""Check that identifier/identify_l1.py gives the same statements as the scripts used in the tests (or_common on da/DAY).
usage: id_check.py DAY_DIR OUT.json     OUT.json = output of identify_l1.py --day-dir DAY_DIR"""
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from or_common import ROOT, load_day, statements  # noqa: E402

day, out = sys.argv[1], json.loads(Path(sys.argv[2]).read_text())
gb = joblib.load(ROOT / "order2" / "l1_model.joblib")
ref = statements(load_day(f"da/{day}"), lambda F: gb.predict_proba(F)[:, 1])
same = diff = 0
for p in out["points"]:
    a, b = sorted(ref[p["frame"]]), p["statements"]
    if a == b:
        same += 1
    else:
        diff += 1
        print("  differs:", p["frame"], "tested scripts", a, "| program", b, "| pictures", p["pictures"])
print(f"{day}: time points compared {same + diff}, identical {same}, different {diff}")
