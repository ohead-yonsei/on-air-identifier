"""AI commentary demo with order (run 3), scoring. usage: cl6_score_order.py DAY     Standard library only. See night/DESIGN_KO.md.
Scored only where the strip was on screen at that time point (the system never saw it):
  numbers        as in cl3_score.py: right = in the strip's first three
  leader claims  sentences in leader mode that name a leader: right = that number is the strip's first
  follow claims  sentences that say the others follow the leader: right = the leader is ahead of every other number said
                 (a number not on the strip counts as behind)
Also the identifier's own leader and order at every strip time point of the day (a further day for leader/RESULT_KO.md)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "live_commentary" / sys.argv[1]
tot = {"points": 0, "leader_mode": 0, "leader_given": 0, "points_with_strip": 0, "numbers": 0, "numbers_right": 0, "points_with_wrong_number": 0,
       "leader_claims": 0, "leader_claims_right": 0, "leader_claims_second": 0, "leader_claims_third": 0, "leader_claims_other": 0, "leader_claims_no_reference": 0,
       "follow_claims": 0, "follow_claims_right": 0, "sources": {"model": 0, "model_retry": 0, "template": 0}}
ident = {"strip_points": 0, "leader_said": 0, "leader_right": 0, "leader_in_first_two": 0, "pairs_said": 0, "pairs_right": 0, "numbers": 0, "numbers_right": 0, "points_good": 0}
races, wrong = {}, []
for f in sorted((O / "commentary_v4").glob("R??.json")):
    d = json.loads(f.read_text())
    per = {"points": len(d["rows"]), "leader_mode": 0, "leader_claims": 0, "leader_claims_right": 0, "numbers": 0, "numbers_right": 0}
    for r in d["rows"]:
        tot["points"] += 1
        tot["sources"][r["source"]] += 1
        if r["mode"] != "leader":
            continue
        tot["leader_mode"] += 1
        per["leader_mode"] += 1
        tot["leader_given"] += r["leader_given"] is not None
        S = r["strip"]
        if not S:
            tot["leader_claims_no_reference"] += len(r["leader_claims"])
            continue
        tot["points_with_strip"] += 1
        right = [n for n in r["said_numbers"] if n in S[:3]]
        tot["numbers"] += len(r["said_numbers"])
        tot["numbers_right"] += len(right)
        per["numbers"] += len(r["said_numbers"])
        per["numbers_right"] += len(right)
        tot["points_with_wrong_number"] += len(right) < len(r["said_numbers"])
        for n in r["leader_claims"]:
            tot["leader_claims"] += 1
            per["leader_claims"] += 1
            key = "right" if n == S[0] else "second" if n == S[1] else "third" if n == S[2] else "other"
            tot["leader_claims_" + key] += 1
            per["leader_claims_right"] += n == S[0]
            if n != S[0]:
                wrong.append({"race": d["race"], "elapsed": r["elapsed"], "line": r["line"], "claimed": n, "strip": S})
        if r["says_follow"] and r["leader_claims"]:
            pos = {n: i for i, n in enumerate(S)}
            lead = r["leader_claims"][0]
            others = [n for n in r["said_numbers"] if n != lead]
            if others:
                tot["follow_claims"] += 1
                tot["follow_claims_right"] += all(pos.get(lead, 99) < pos.get(n, 99) for n in others)
    races[d["race"]] = per
for f in sorted((O / "results_f2s").glob("R??.json")):
    for p in json.loads(f.read_text())["points"]:
        S = p.get("strip")
        if not S:
            continue
        ident["strip_points"] += 1
        c = sum(n in S[:3] for n in p["statements"])
        ident["numbers"] += len(p["statements"])
        ident["numbers_right"] += c
        ident["points_good"] += c > 0
        if p.get("leader") is not None:
            ident["leader_said"] += 1
            ident["leader_right"] += p["leader"] == S[0]
            ident["leader_in_first_two"] += p["leader"] in S[:2]
        pos = {n: i for i, n in enumerate(S)}
        for o in p.get("order", []):
            if o["ahead"] in pos or o["behind"] in pos:
                ident["pairs_said"] += 1
                ident["pairs_right"] += pos.get(o["ahead"], 99) < pos.get(o["behind"], 99)
out = {"day": sys.argv[1], "commentary": tot, "identifier_on_strip_points": ident, "by_race": races, "wrong_leader_sentences": wrong}
(O / "score_v4.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({"commentary": tot, "identifier": ident}, ensure_ascii=False, indent=1))
for w in wrong:
    print("  wrong leader:", w)
