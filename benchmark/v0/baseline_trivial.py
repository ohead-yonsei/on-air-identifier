"""ON-AIR draft benchmark v0: two reference lines that need no model.
usage: python3 benchmark/v0/baseline_trivial.py [--official]        (standard library only)

  confirmed_only   never names a horse: confirmed information only (0 % hallucination, 0 % coverage; shows why both are counted)
  random_three     names three runners of the race at random (seed 20261118; what guessing gives)"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
meta = json.loads((HERE / "races.json").read_text())["races"]
OFFICIAL = "--official" in sys.argv          # the official track (items_official.jsonl); output names get the ending _official
items = [json.loads(l) for l in (HERE / ("items_official.jsonl" if OFFICIAL else "items.jsonl")).read_text().splitlines()]
rng = random.Random(20261118)
a, b = [], []
for it in items:
    r = meta[it["race_key"]]
    a.append({"id": it["id"], "text": f"서울 {r['race']}경주 {r['distance_m']}m, 출발 후 {int(it['seconds_since_start'])}초가 지났습니다."})
    three = rng.sample(r["runners"], 3)
    b.append({"id": it["id"], "text": f"선두는 {three[0]}번, 선두권에 {three[1]}번, {three[2]}번도 있습니다."})
(HERE / "baselines").mkdir(exist_ok=True)
for name, rows in (("confirmed_only", a), ("random_three", b)):
    (HERE / "baselines" / f"{name}{'_official' if OFFICIAL else ''}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(name, len(rows))
