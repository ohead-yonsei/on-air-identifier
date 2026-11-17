"""Score leader claims per condition against the strip. See commentary/DESIGN_KO.md.

usage: python t2_score_commentary.py test
"""
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "commentary"
split = sys.argv[1]
rows = {}
for line in (C / f"raw_{split}.jsonl").read_text().splitlines():
    r = json.loads(line)
    if not r["failure"]:
        rows[(r["frame"], r["condition"])] = r
frames = sorted({f for f, _ in rows})
CONDS = ["A", "R", "F", "B"]
complete = [f for f in frames if all((f, c) in rows for c in CONDS)]
wrong3 = {c: {} for c in CONDS}
res = {}
for c in CONDS:
    k = {"claims": 0, "wrong_top3": 0, "wrong_top5": 0, "frames_with_wrong": 0, "frames_silent": 0, "frames_with_correct": 0,
         "claims_equal_to_facts": 0, "frames_answer_equals_facts": 0, "ms": 0.0}
    for f in complete:
        r = rows[(f, c)]
        said = list(dict.fromkeys(r["leaders"]))  # drop repeats within one answer
        B = r["strip"]
        w3 = sum(n not in B[:3] for n in said)
        wrong3[c][f] = w3
        k["claims"] += len(said)
        k["wrong_top3"] += w3
        k["wrong_top5"] += sum(n not in B for n in said)
        k["frames_with_wrong"] += w3 > 0
        k["frames_silent"] += not said
        k["frames_with_correct"] += any(n in B[:3] for n in said)
        k["claims_equal_to_facts"] += sum(n in r["facts"] for n in said)
        k["frames_answer_equals_facts"] += sorted(said) == sorted(r["facts"])
        k["ms"] += r["ms"]
    k["wrong_rate_top3"] = k["wrong_top3"] / k["claims"] if k["claims"] else None
    k["wrong_rate_top5"] = k["wrong_top5"] / k["claims"] if k["claims"] else None
    k["ms_per_call"] = k["ms"] / max(1, len(complete))
    res[c] = k


def paired(a, b):
    fewer = sum(wrong3[b][f] < wrong3[a][f] for f in complete)
    more = sum(wrong3[b][f] > wrong3[a][f] for f in complete)
    n, m = fewer + more, min(fewer, more)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0
    return {"frames_with_fewer_wrong": fewer, "frames_with_more_wrong": more, "sign_test_p": p}


facts_frames = sum(bool(rows[(f, "F")]["facts"]) for f in complete)
facts_n = sum(len(rows[(f, "F")]["facts"]) for f in complete)
facts_ok = sum(n in rows[(f, "F")]["strip"][:3] for f in complete for n in rows[(f, "F")]["facts"])
primary = paired("A", "B")
out = {"split": split, "frames_complete": len(complete), "conditions": res, "primary_A_vs_B": primary,
       "hypothesis_supported": bool(primary["frames_with_fewer_wrong"] > primary["frames_with_more_wrong"] and primary["sign_test_p"] < 0.05),
       "A_vs_R": paired("A", "R"), "A_vs_F": paired("A", "F"), "F_vs_B": paired("F", "B"),
       "identifier_facts": {"frames_with_facts": facts_frames, "facts": facts_n, "facts_in_top3": facts_ok}}
(C / f"score_{split}.json").write_text(json.dumps(out, indent=1))
print(f"split {split}: {len(complete)} frames with all four answers | identifier facts in {facts_frames} frames, {facts_ok}/{facts_n} in strip top 3")
print("cond  claims  wrong(top3)  rate    wrong(top5)  frames w/ wrong  silent frames  frames w/ a correct number  claims that repeat a fact  answer == facts")
for c in CONDS:
    k = res[c]
    rate = "n/a" if k["wrong_rate_top3"] is None else f"{k['wrong_rate_top3']:.3f}"
    print(f"{c}     {k['claims']:4d}    {k['wrong_top3']:4d}        {rate}   {k['wrong_top5']:4d}         {k['frames_with_wrong']:3d}              {k['frames_silent']:3d}            {k['frames_with_correct']:3d}"
          f"                         {k['claims_equal_to_facts']:4d}                       {k['frames_answer_equals_facts']:3d}")
for name, p in (("A -> R", out["A_vs_R"]), ("A -> F", out["A_vs_F"]), ("A -> B (primary)", primary), ("F -> B", out["F_vs_B"])):
    print(f"  {name:18s} fewer wrong in {p['frames_with_fewer_wrong']:2d} frames / more wrong in {p['frames_with_more_wrong']:2d}  p={p['sign_test_p']:.4f}")
print("hypothesis supported:", out["hypothesis_supported"])
