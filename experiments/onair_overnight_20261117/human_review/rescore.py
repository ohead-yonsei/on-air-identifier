"""Re-score the cap-region comparison against the human review of 2026-11-17.

Reads Codex's outputs without modifying them:
  review/USER_REVIEW.csv           draft cap numbers
  results/condition_comparison.csv one row per sample x region method x classifier
  human_review/review_paste_20261117.json   the reviewer's answers pasted from REVIEW.html
  human_review/review_addendum_20261117.json  answers given later in chat (R07)

Writes human_review/human_labels.json, rescore_summary.json and rescore_rows.csv.

Human label per sample:
  ok      -> the draft number is confirmed (a draft of "unclear" stays unknown)
  fix     -> the reviewer's number replaces the draft
  unknown -> excluded from accuracy
  absent  -> not reviewed; excluded from accuracy
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "human_review"
METHODS = ["R0_27", "R0_35", "R1_pose", "R2_cluster", "U_manual"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


paste = json.loads((OUT / "review_paste_20261117.json").read_text(encoding="utf-8"))
addendum_path = OUT / "review_addendum_20261117.json"
if addendum_path.exists():
    # Later answers given in chat; they only fill samples the first paste left unreviewed.
    for sid, ans in json.loads(addendum_path.read_text(encoding="utf-8"))["caps"].items():
        assert sid not in paste["caps"], sid
        paste["caps"][sid] = ans
drafts = {r["sample_id"]: r for r in read_csv(ROOT / "review" / "USER_REVIEW.csv")}

labels = {}
for sid, d in drafts.items():
    ans = paste["caps"].get(sid)
    draft = d["draft_number"]
    if ans is None:
        human, how = None, "not_reviewed"
    elif ans["status"] == "ok":
        human, how = (None, "confirmed_unclear") if draft == "unclear" else (draft, "confirmed")
    elif ans["status"] == "fix":
        human, how = ans["number"], "corrected"
    else:
        human, how = None, "marked_unknown"
    labels[sid] = {"race": d["race"], "pts": d["pts"], "draft_number": draft,
                   "draft_colour": d["draft_colour"], "human_number": human, "review": how}

(OUT / "human_labels.json").write_text(json.dumps({
    "source": "REVIEW.html answers pasted by the user on 2026-11-17",
    "scope": "cap number (colour) only; cap box position, nose point and running order are not confirmed by this review",
    "counts": dict(Counter(v["review"] for v in labels.values())),
    "labels": labels,
}, ensure_ascii=False, indent=2), encoding="utf-8")

rows = read_csv(ROOT / "results" / "condition_comparison.csv")
scored = []
for r in rows:
    if r["split"] == "human_check":
        truth = r["reference_number"]
    else:
        truth = labels[r["sample_id"]]["human_number"]
    if truth is None:
        continue
    pred = r["predicted_number"]
    verdict = "abstain" if pred == "0" else ("correct" if pred == truth else "wrong")
    scored.append({**{k: r[k] for k in ("sample_id", "race", "split", "method", "classifier", "view", "size_group")},
                   "human_number": truth, "predicted_number": pred, "verdict": verdict,
                   "draft_number": r["reference_number"]})

with open(OUT / "rescore_rows.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(scored[0]))
    w.writeheader()
    w.writerows(scored)


def tally(sel):
    c = Counter(s["verdict"] for s in sel)
    return {"n": len(sel), "correct": c["correct"], "wrong": c["wrong"], "abstain": c["abstain"]}


def table(split_filter, group=None):
    out = {}
    for clf in ("vlm", "hsv"):
        for m in METHODS:
            sel = [s for s in scored if split_filter(s) and s["classifier"] == clf and s["method"] == m
                   and (group is None or group(s))]
            out[f"{clf}|{m}"] = tally(sel)
    return out


def paired(split_filter, clf, a="R0_35", b="U_manual"):
    by = defaultdict(dict)
    for s in scored:
        if split_filter(s) and s["classifier"] == clf and s["method"] in (a, b):
            by[s["sample_id"]][s["method"]] = s["verdict"]
    c = Counter()
    detail = []
    for sid, v in sorted(by.items()):
        if a in v and b in v:
            ka, kb = v[a] == "correct", v[b] == "correct"
            key = "both_correct" if ka and kb else "improved" if kb else "worsened" if ka else "both_not_correct"
            c[key] += 1
            if key in ("improved", "worsened"):
                detail.append({"sample_id": sid, "change": key, a: v[a], b: v[b]})
    return {"counts": dict(c), "changed": detail}


is_eval = lambda s: s["split"] == "evaluation"
is_dev = lambda s: s["split"] == "development"
is_sep = lambda s: s["split"] in ("evaluation", "development")
is_oct = lambda s: s["split"] == "human_check"
is_all = lambda s: True

summary = {
    "label_review": dict(Counter(v["review"] for v in labels.values())),
    "samples_with_human_number": {
        "evaluation": len({s["sample_id"] for s in scored if is_eval(s)}),
        "development": len({s["sample_id"] for s in scored if is_dev(s)}),
        "oct10_human_check": len({s["sample_id"] for s in scored if is_oct(s)}),
    },
    "evaluation": table(is_eval),
    "development": table(is_dev),
    "sep06_all_reviewed": table(is_sep),
    "oct10_human_check_uncorrected": table(is_oct),
    "all_human_labels": table(is_all),
    "paired_R0_35_to_U_vlm": {"evaluation": paired(is_eval, "vlm"), "sep06_all": paired(is_sep, "vlm")},
    "paired_R0_35_to_U_hsv": {"evaluation": paired(is_eval, "hsv"), "sep06_all": paired(is_sep, "hsv")},
    "evaluation_by_race_vlm": {
        race: {m: tally([s for s in scored if is_eval(s) and s["race"] == race and s["classifier"] == "vlm"
                         and s["method"] == m]) for m in ("R0_35", "U_manual")}
        for race in sorted({s["race"] for s in scored if is_eval(s)})
    },
    "sep06_by_view_vlm": {
        view: {m: tally([s for s in scored if is_sep(s) and s["view"] == view and s["classifier"] == "vlm"
                         and s["method"] == m]) for m in METHODS}
        for view in sorted({s["view"] for s in scored if is_sep(s)})
    },
    "sep06_by_size_vlm": {
        size: {m: tally([s for s in scored if is_sep(s) and s["size_group"] == size and s["classifier"] == "vlm"
                         and s["method"] == m]) for m in METHODS}
        for size in ("small", "medium", "large")
    },
    "sep06_by_true_number_vlm_U": {
        num: tally([s for s in scored if is_sep(s) and s["human_number"] == num and s["classifier"] == "vlm"
                    and s["method"] == "U_manual"])
        for num in sorted({s["human_number"] for s in scored if is_sep(s)}, key=int)
    },
    "sep06_by_true_number_vlm_R0_35": {
        num: tally([s for s in scored if is_sep(s) and s["human_number"] == num and s["classifier"] == "vlm"
                    and s["method"] == "R0_35"])
        for num in sorted({s["human_number"] for s in scored if is_sep(s)}, key=int)
    },
}
(OUT / "rescore_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def show(title, t):
    print(f"\n{title}")
    for clf in ("vlm", "hsv"):
        for m in METHODS:
            v = t[f"{clf}|{m}"]
            print(f"  {clf:3s} {m:11s} correct {v['correct']:2d} / wrong {v['wrong']:2d} / abstain {v['abstain']:2d}  (n={v['n']})")


print("label review:", summary["label_review"])
print("samples with a human number:", summary["samples_with_human_number"])
show("EVALUATION races (R1 R2 R4 R8 R9 R10)", summary["evaluation"])
show("DEVELOPMENT races (R3 R5 R6 R7)", summary["development"])
show("ALL reviewed Sep-06", summary["sep06_all_reviewed"])
show("Oct-10 six human-confirmed (uncorrected run)", summary["oct10_human_check_uncorrected"])
print("\npaired R0_35 -> U (vlm):", json.dumps(summary["paired_R0_35_to_U_vlm"], ensure_ascii=False))
print("\nby race (vlm):", json.dumps(summary["evaluation_by_race_vlm"]))
print("\nby view (vlm):", json.dumps(summary["sep06_by_view_vlm"]))
print("\nby size (vlm):", json.dumps(summary["sep06_by_size_vlm"]))
print("\nby true number, vlm U:", json.dumps(summary["sep06_by_true_number_vlm_U"]))
print("by true number, vlm R0_35:", json.dumps(summary["sep06_by_true_number_vlm_R0_35"]))
