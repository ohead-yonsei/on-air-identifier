"""caps2: two benchmark v0 prediction files compared item by item (same items, same answers), on subsets of the test days.
usage: python3 h6_bench_compare.py OUT.json CURRENT.jsonl NEW.jsonl       (standard library only; run from anywhere)

Subsets of the test split:
  clean   2026-09-12, 09-05, 08-30   no cap label of the new classifier was mined from these days
  mined   2026-10-05, 10-03, 09-19   days that gave mined cap labels (the new classifier has seen caps of these races)
  all     the six test days (874 items)
Metrics as benchmark/v0/score.py. Differences new minus current: bootstrap over races, 2000 resamples, seed 20261118.
The rule for changing the program default is in caps2/DESIGN_BENCH_KO.md and is applied to the clean subset."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark" / "v0"))
import score

SUBSETS = {"clean": ("20260912", "20260905", "20260830"), "mined": ("20261005", "20261003", "20260919"),
           "all": ("20260912", "20260905", "20260830", "20261005", "20261003", "20260919")}
KEYS = ("leader_right_share", "leader_wrong_share", "hallucination", "A", "B", "right_and_clean", "leader")


def diff(new, old, key, n=2000, seed=20261118):
    races = sorted({r["race"] for r in new})
    bn = {k: [r for r in new if r["race"] == k] for k in races}
    bo = {k: [r for r in old if r["race"] == k] for k in races}
    rng, vals = random.Random(seed), []
    for _ in range(n):
        pk = rng.choices(races, k=len(races))
        a, b = score.summarise([x for k in pk for x in bn[k]])[key], score.summarise([x for k in pk for x in bo[k]])[key]
        if a is not None and b is not None:
            vals.append(round(a - b, 1))
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def main():
    out, cur, new = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    rc, rn = score.rows(cur, "test"), score.rows(new, "test")
    res = {"current": cur, "new": new}
    for name, days in SUBSETS.items():
        a, b = [r for r in rc if r["id"][:8] in days], [r for r in rn if r["id"][:8] in days]
        sa, sb = score.summarise(a), score.summarise(b)
        d = {k: {"current": sa[k], "new": sb[k], "difference": None if sa[k] is None or sb[k] is None else round(sb[k] - sa[k], 1), "interval": diff(b, a, k)} for k in KEYS}
        res[name] = {"items": len(a), "races": len({r["race"] for r in a}), "metrics": d, "current_counts": {k: sa[k] for k in ("numbers", "numbers_right", "items_leader_right", "items_leader_wrong", "items_with_wrong_number", "items_with_right_number")},
                     "new_counts": {k: sb[k] for k in ("numbers", "numbers_right", "items_leader_right", "items_leader_wrong", "items_with_wrong_number", "items_with_right_number")}}
        print(f"== {name}: {len(a)} items, {res[name]['races']} races")
        for k in KEYS:
            v = d[k]
            print(f"   {k:20s} current {v['current']}%  new {v['new']}%  difference {v['difference']} {v['interval']}")
    m = res["clean"]["metrics"]
    no_harm = m["hallucination"]["difference"] <= 1.0 and m["A"]["difference"] >= -1.0 and m["leader_wrong_share"]["difference"] <= 1.0
    gain = m["B"]["interval"][0] > 0 or m["leader_right_share"]["interval"][0] > 0
    res["rule"] = {"no_harm": bool(no_harm), "gain": bool(gain), "change_the_default": bool(no_harm and gain)}
    print("rule on the clean days:", res["rule"])
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
