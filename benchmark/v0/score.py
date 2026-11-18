"""ON-AIR draft benchmark v0: scorer.
usage: python3 benchmark/v0/score.py PREDICTIONS.jsonl [--split test|dev|all] [--out SCORE.json]     (standard library only)

PREDICTIONS.jsonl: one line per item, {"id": ..., "text": "한국어 중계 한 문장"}.
Horses are referred to by number, written like "4번". Optional fields override what is read from the text:
"numbers": [..] (the numbers the sentence puts in the leading group), "leader": n or null (the number it calls the leader).
Items without a line count as "said nothing".

What is counted (the answer is the running-order strip at that moment):
  wrong number      a stated number that is not among the strip's first three
  hallucination     share of items whose sentence contains at least one wrong number             (lower is better)
  A (precision)     stated numbers that are among the first three / all stated numbers           (higher is better)
  B (coverage)      items with at least one right number / all items                             (higher is better; says how much was told)
  leader            leader claims that name the strip's first / all leader claims, and how many items carry a leader claim
  not in the race   stated numbers that do not run in this race at all
  fourth or fifth   wrong numbers that are fourth or fifth on the strip (near misses)
Intervals: bootstrap over races, 2000 resamples, seed 20261118."""
import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUMBER = re.compile(r"(\d+)\s*번")
LEADER = re.compile(r"선두는\s*(?:[가-힣 ]{0,12}모자\s*)?(\d+)\s*번|선두(?:에|로)\s*(?:[가-힣 ]{0,12}모자\s*)?(\d+)\s*번|(\d+)\s*번이\s*선두(?!권)|(\d+)\s*번이\s*(?:맨\s*)?앞(?:장|서)|(\d+)\s*번이\s*(?:경주를\s*)?이끌")


def claims(pred):
    text = (pred or {}).get("text") or ""
    nums = pred["numbers"] if pred and "numbers" in pred else sorted({int(n) for n in NUMBER.findall(text)})
    if pred and "leader" in pred:
        lead = [pred["leader"]] if pred["leader"] is not None else []
    else:
        lead = sorted({int(g) for m in LEADER.finditer(text) for g in m.groups() if g})
    return sorted(set(nums) | set(lead)), lead


def rows(pred_path, split):
    items = [json.loads(l) for l in (HERE / "items.jsonl").read_text().splitlines()]
    ans = {a["id"]: a["strip"] for a in map(json.loads, (HERE / "answers.jsonl").read_text().splitlines())}
    races = json.loads((HERE / "races.json").read_text())["races"]
    pred = {p["id"]: p for p in map(json.loads, Path(pred_path).read_text().splitlines()) if p}
    out = []
    for it in items:
        if split != "all" and it["split"] != split:
            continue
        strip, runners = ans[it["id"]], set(races[it["race_key"]]["runners"])
        nums, lead = claims(pred.get(it["id"]))
        out.append({"id": it["id"], "race": it["race_key"], "said": len(nums), "right": sum(n in strip[:3] for n in nums),
                    "near": sum(n in strip[3:] for n in nums), "absent": sum(n not in runners for n in nums) if runners else 0,
                    "lead": len(lead), "lead_right": sum(n == strip[0] for n in lead), "lead_second": sum(n == strip[1] for n in lead),
                    "answered": it["id"] in pred})
    return out


def summarise(rs):
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    said, n = sum(r["said"] for r in rs), len(rs)
    s = {"items": n, "items_with_a_number": sum(r["said"] > 0 for r in rs), "numbers": said, "numbers_right": sum(r["right"] for r in rs),
         "numbers_fourth_or_fifth": sum(r["near"] for r in rs), "numbers_not_in_race": sum(r["absent"] for r in rs),
         "items_with_wrong_number": sum(r["said"] > r["right"] for r in rs), "items_with_right_number": sum(r["right"] > 0 for r in rs),
         "leader_claims": sum(r["lead"] for r in rs), "leader_right": sum(r["lead_right"] for r in rs), "leader_second": sum(r["lead_second"] for r in rs),
         "items_with_leader_claim": sum(r["lead"] > 0 for r in rs)}
    s.update(hallucination=pct(s["items_with_wrong_number"], n), A=pct(s["numbers_right"], said), B=pct(s["items_with_right_number"], n),
             leader=pct(s["leader_right"], s["leader_claims"]), leader_coverage=pct(s["items_with_leader_claim"], n))
    return s


def interval(rs, key, n=2000, seed=20261118):
    races = sorted({r["race"] for r in rs})
    by = {k: [r for r in rs if r["race"] == k] for k in races}
    rng, vals = random.Random(seed), []
    for _ in range(n):
        v = summarise([x for k in rng.choices(races, k=len(races)) for x in by[k]])[key]
        if v is not None:
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--split", default="test", choices=("test", "dev", "all"))
    ap.add_argument("--out")
    ap.add_argument("--only-answered", action="store_true", help="score only the items that have a line (for a run that is not finished)")
    a = ap.parse_args()
    rs = rows(a.predictions, a.split)
    if a.only_answered:
        rs = [r for r in rs if r["answered"]]
    s = summarise(rs)
    s["interval"] = {k: interval(rs, k) for k in ("hallucination", "A", "B", "leader")}
    s.update(split=a.split, races=len({r["race"] for r in rs}), predictions=a.predictions, items_without_a_line=sum(not r["answered"] for r in rs))
    print(f"{a.predictions}  [{a.split}: {s['races']} races, {s['items']} items, {s['items_without_a_line']} without a line]")
    print(f"  hallucination (items with a wrong number)  {s['hallucination']}%  ({s['items_with_wrong_number']}/{s['items']})  {s['interval']['hallucination']}")
    print(f"  A  right numbers / stated numbers          {s['A']}%  ({s['numbers_right']}/{s['numbers']})  {s['interval']['A']}")
    print(f"  B  items with a right number               {s['B']}%  ({s['items_with_right_number']}/{s['items']})  {s['interval']['B']}")
    print(f"  leader claims right                        {s['leader']}%  ({s['leader_right']}/{s['leader_claims']})  {s['interval']['leader']}   in {s['leader_coverage']}% of items; second {s['leader_second']}")
    print(f"  wrong numbers: fourth or fifth {s['numbers_fourth_or_fifth']}, not in the race {s['numbers_not_in_race']}, other {s['numbers'] - s['numbers_right'] - s['numbers_fourth_or_fifth'] - s['numbers_not_in_race']}")
    if a.out:
        Path(a.out).write_text(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
