"""Front-three scoring against the broadcaster's running-order strip. See leaders/DESIGN_KO.md.

Q1/Q2 use human labels only. Q3 uses identifier outputs in leaders/ident_out/<frame>.json.
Writes leaders/summary.json and leaders/frames.json.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
L = ROOT / "leaders"
strip = json.loads((L / "strip_order.json").read_text())["order"]

human = defaultdict(list)  # frame -> [(number, box)]
for ev in (ROOT / "results" / "evidence.json", ROOT / "batch2" / "results" / "evidence.json", ROOT / "batch2b" / "results" / "evidence.json"):
    for r in json.loads(ev.read_text()):
        if r["human_number"] and not r["duplicate_of"] and r["frame"] in strip:
            human[r["frame"]].append((r["human_number"], r["person_box"]))

bottom = lambda box: box[3]
q1 = Counter()
q2 = Counter()
q3 = Counter()
per_frame = []
for f in sorted(strip):
    B = strip[f]
    rank = {n: i for i, n in enumerate(B)}
    rec = {"frame": f, "strip": B}

    # ---- Q1: vertical rule vs strip, human numbers only ----
    hs = [(n, box) for n, box in human[f] if n in rank]
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            (a, ba), (b, bb) = hs[i], hs[j]
            rule_a_ahead = bottom(ba) > bottom(bb)
            strip_a_ahead = rank[a] < rank[b]
            q1["pairs"] += 1
            q1["pairs_agree"] += rule_a_ahead == strip_a_ahead
    if len(hs) >= 2:
        rule_first = max(hs, key=lambda x: bottom(x[1]))[0]
        strip_first = min(hs, key=lambda x: rank[x[0]])[0]
        q1["frames_with_2plus"] += 1
        q1["front_of_labelled_agree"] += rule_first == strip_first
        rec["human_in_strip_by_rule"] = [n for n, _ in sorted(hs, key=lambda x: -bottom(x[1]))]

    # ---- Q2: are the strip's top three boxed and labelled? ----
    nums = {n for n, _ in human[f]}
    for k in range(3):
        q2[f"top{k + 1}_total"] += 1
        q2[f"top{k + 1}_labelled"] += B[k] in nums

    # ---- Q3: identifier ----
    p = L / "ident_out" / f"{f}.json"
    if p.exists():
        riders = sorted(json.loads(p.read_text())["riders"], key=lambda r: -bottom(r["box"]))
        front = riders[:3]
        rec["system_front3"] = [(r["no"], r["state"]) for r in front]
        q3["frames"] += 1
        if front:
            lead = front[0]
            st = "sure" if lead["state"] == "sure" else "not_sure"
            q3[f"leader_{st}"] += 1
            q3[f"leader_{st}_strict"] += lead["no"] == B[0]
            q3[f"leader_{st}_lenient"] += lead["no"] in B[:3]
        for r in front:
            if r["state"] == "sure":
                q3["front3_sure"] += 1
                q3["front3_sure_in_top3"] += r["no"] in B[:3]
                q3["front3_sure_in_top5"] += r["no"] in B
            elif r["no"]:
                q3["front3_guess"] += 1
                q3["front3_guess_in_top3"] += r["no"] in B[:3]
                q3["front3_guess_in_top5"] += r["no"] in B
        named = {r["no"] for r in front if r["no"]}
        named_sure = {r["no"] for r in front if r["state"] == "sure"}
        for k in range(3):
            q3[f"strip_top{k + 1}_named_in_front3"] += B[k] in named
            q3[f"strip_top{k + 1}_named_sure_in_front3"] += B[k] in named_sure
        # which human-confirmed horse is in the identifier's front box? (separates box/order errors from number errors)
        if front:
            hb = [n for n, box in human[f] if abs(box[0] - front[0]["box"][0]) < 2 and abs(box[3] - front[0]["box"][3]) < 2]
            if hb:
                q3["front_box_has_human_number"] += 1
                q3["front_box_human_is_strip_first"] += hb[0] == B[0]
                q3["front_box_human_in_strip_top3"] += hb[0] in B[:3]
                q3["front_box_number_matches_human"] += front[0]["no"] == hb[0]
                rec["front_box_human"] = hb[0]
    per_frame.append(rec)

summary = {"frames_with_strip": len(strip), "Q1_vertical_rule": dict(q1), "Q2_coverage": dict(q2), "Q3_identifier": dict(q3)}
(L / "summary.json").write_text(json.dumps(summary, indent=1))
(L / "frames.json").write_text(json.dumps(per_frame, indent=1))
r = lambda a, b: f"{a}/{b}" + (f" ({a / b:.2f})" if b else "")
print("frames with a strip:", len(strip))
print("Q1 vertical rule vs strip (human numbers): pairs agree", r(q1["pairs_agree"], q1["pairs"]),
      "| front-most labelled agrees", r(q1["front_of_labelled_agree"], q1["frames_with_2plus"]))
print("Q2 strip top-3 boxed and labelled:", " ".join(f"#{k}: " + r(q2[f"top{k}_labelled"], q2[f"top{k}_total"]) for k in (1, 2, 3)))
if q3["frames"]:
    print("Q3 identifier, frames:", q3["frames"])
    print("  front box marked sure:", q3["leader_sure"], "| equals strip #1", r(q3["leader_sure_strict"], q3["leader_sure"]), "| in strip top 3", r(q3["leader_sure_lenient"], q3["leader_sure"]))
    print("  front box not sure  :", q3["leader_not_sure"], "| equals strip #1", r(q3["leader_not_sure_strict"], q3["leader_not_sure"]), "| in strip top 3", r(q3["leader_not_sure_lenient"], q3["leader_not_sure"]))
    print("  front-three sure answers:", q3["front3_sure"], "| in strip top 3", r(q3["front3_sure_in_top3"], q3["front3_sure"]), "| in strip top 5", r(q3["front3_sure_in_top5"], q3["front3_sure"]))
    print("  front-three guesses     :", q3["front3_guess"], "| in strip top 3", r(q3["front3_guess_in_top3"], q3["front3_guess"]), "| in strip top 5", r(q3["front3_guess_in_top5"], q3["front3_guess"]))
    print("  strip #1/#2/#3 named among the identifier's front three:", " ".join(r(q3[f"strip_top{k}_named_in_front3"], q3["frames"]) for k in (1, 2, 3)),
          "| as sure:", " ".join(r(q3[f"strip_top{k}_named_sure_in_front3"], q3["frames"]) for k in (1, 2, 3)))
    print("  front box has a human number:", q3["front_box_has_human_number"], "| that horse is strip #1", r(q3["front_box_human_is_strip_first"], q3["front_box_has_human_number"]),
          "| in strip top 3", r(q3["front_box_human_in_strip_top3"], q3["front_box_has_human_number"]),
          "| identifier number matches human", r(q3["front_box_number_matches_human"], q3["front_box_has_human_number"]))
