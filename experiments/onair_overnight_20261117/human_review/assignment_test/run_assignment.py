"""Within-frame one-to-one assignment test on saved HSV cap scores. See DESIGN_KO.md.

Uses only saved score vectors (results/regions_*.json) and the human labels. Pure stdlib.
"""
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
METHODS = ["R0_27", "R0_35", "R1_pose", "R2_cluster", "U_manual"]
NUMBERS = list(range(1, 12))

labels = json.loads((ROOT / "human_review" / "human_labels.json").read_text(encoding="utf-8"))["labels"]
recs = (json.loads((ROOT / "results" / "regions_evaluation.json").read_text())
        + json.loads((ROOT / "results" / "regions_development_human_check.json").read_text()))


def truth(r):
    if r["split"] == "human_check":
        return str(r["reference_number"])
    return labels[r["sample_id"]]["human_number"]


def scores(r):
    s = r["hsv"].get("all_scores")
    return list(s) if s else None  # None when the region could not be produced


def independent(s):
    if s is None or max(s) <= 0:
        return 0
    return NUMBERS[max(range(len(s)), key=lambda i: (s[i], -i))]  # ties -> lowest number


def assign(score_list):
    """Distinct numbers for the samples that have scores, maximising the score sum.
    Returns (numbers, tie_flag). Brute force: at most 11P6 = 332,640 permutations."""
    idx = [i for i, s in enumerate(score_list) if s is not None]
    out = [0] * len(score_list)
    if not idx:
        return out, False
    best, best_perm, ties = -1.0, None, 0
    for perm in itertools.permutations(NUMBERS, len(idx)):
        total = sum(score_list[i][n - 1] for i, n in zip(idx, perm))
        if total > best + 1e-12:
            best, best_perm, ties = total, perm, 0
        elif abs(total - best) <= 1e-12:
            ties += 1  # permutations are generated in lexicographic order, so the first one is kept
    for i, n in zip(idx, best_perm):
        out[i] = n if score_list[i][n - 1] > 0 else 0
    return out, ties > 0


rows = []
tie_frames = defaultdict(list)
for method in METHODS:
    frames = defaultdict(list)
    for r in recs:
        if r["method"] == method:
            frames[r["frame"]].append(r)
    for frame, items in sorted(frames.items()):
        items.sort(key=lambda r: r["sample_id"])
        sl = [scores(r) for r in items]
        a1, tie = assign(sl)
        if tie:
            tie_frames[method].append(frame)
        for r, s, a in zip(items, sl, a1):
            t = truth(r)
            rows.append({
                "sample_id": r["sample_id"], "frame": frame, "race": r["race"], "split": r["split"],
                "method": method, "human_number": t or "",
                "I0": r["hsv"]["number"], "I1": independent(s), "A1": a,
                "scores": "" if s is None else " ".join(f"{x:.3f}" for x in s),
            })


def verdict(pred, t):
    return "abstain" if int(pred) == 0 else ("correct" if str(pred) == t else "wrong")


for r in rows:
    for c in ("I0", "I1", "A1"):
        r[c + "_verdict"] = verdict(r[c], r["human_number"]) if r["human_number"] else ""

with open(HERE / "assignment_rows.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

SUBSETS = {
    "sep06_all_57": lambda r: r["split"] in ("evaluation", "development"),
    "evaluation_34": lambda r: r["split"] == "evaluation",
    "development_23": lambda r: r["split"] == "development",
    "oct10_one_frame_6": lambda r: r["split"] == "human_check",
}
summary = {"ties": {m: v for m, v in tie_frames.items()}, "results": {}}
for name, flt in SUBSETS.items():
    summary["results"][name] = {}
    for method in METHODS:
        sel = [r for r in rows if flt(r) and r["method"] == method and r["human_number"]]
        entry = {"n": len(sel)}
        for c in ("I0", "I1", "A1"):
            k = Counter(r[c + "_verdict"] for r in sel)
            entry[c] = {"correct": k["correct"], "wrong": k["wrong"], "abstain": k["abstain"]}
        ch = Counter()
        changed = []
        for r in sel:
            a, b = r["I1_verdict"] == "correct", r["A1_verdict"] == "correct"
            if b and not a:
                ch["improved"] += 1
            elif a and not b:
                ch["worsened"] += 1
            if r["I1"] != r["A1"]:
                ch["changed_prediction"] += 1
                changed.append({"sample_id": r["sample_id"], "human": r["human_number"], "I1": r["I1"], "A1": r["A1"]})
        entry["I1_to_A1"] = dict(ch)
        entry["changed"] = changed
        summary["results"][name][method] = entry

(HERE / "assignment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

for name in SUBSETS:
    print(f"\n{name}")
    for method in METHODS:
        e = summary["results"][name][method]
        f3 = lambda d: f"{d['correct']:2d}/{d['wrong']:2d}/{d['abstain']:2d}"
        print(f"  {method:11s} n={e['n']:2d}  I0 {f3(e['I0'])}  I1 {f3(e['I1'])}  A1 {f3(e['A1'])}  I1->A1 {e['I1_to_A1']}")
print("\nties:", summary["ties"])
for m in ("U_manual", "R0_35"):
    print(f"\nchanged predictions, {m}, sep06:")
    for c in summary["results"]["sep06_all_57"][m]["changed"]:
        print("  ", c)
    print(f"changed predictions, {m}, oct10:")
    for c in summary["results"]["oct10_one_frame_6"][m]["changed"]:
        print("  ", c)
