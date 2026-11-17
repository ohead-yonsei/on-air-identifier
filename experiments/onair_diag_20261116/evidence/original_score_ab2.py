"""Score the five-arm responses statement by statement (v2: also checks where-on-track / distance claims and counts explicit "cannot tell").
usage: python3 score_ab2.py scenes.json responses.jsonl scored.jsonl [--judge qwen3-vl:8b-instruct]"""
import argparse, json, os, re, csv, collections

JUDGE = """You label the statements of a horse-race commentary. Return JSON only.
RUNNERS: {names}
WHERE THE FIELD IS NOW: it has already passed: {passed}. It has NOT yet reached: {ahead}.
VERIFIED FACTS:
{facts}

COMMENTARY:
{text}

Split the commentary into separate statements, one claim each (split "A leads, followed by B and C" into three). For each give:
- "text": the statement
- "type": one of
    "position" = where a NAMED horse is in the race right now (leads, second, third, at the back, ahead of / behind another horse)
    "named"    = a NAMED horse is doing something now, with no checkable position
    "location" = where the field is on the track now, or how far there is to go (turns, straights, distance marks, metres to go)
    "fact"     = background: conditions, race distance, number of runners, which jockey rides which horse, cap colours
    "abstain"  = says it cannot tell / cannot identify who is where
    "vague"    = talk about the race now with no name and nothing checkable ("they are bunched up", "the pace is strong", "no clear leader")
    "other"    = anything else
- "horse": for position/named: the runner's name exactly as in RUNNERS, else null
- "rank": for position: 1 if the horse leads / is in front, 2 second, 3 third ...; "last" if at the rear; null if no rank is stated
- "ahead_of": for position: name of a runner this horse is said to be in front of, else null
- "loc_ok": for location: true if it agrees with WHERE THE FIELD IS NOW and the VERIFIED FACTS (saying the field approaches the next point not yet reached is fine); false if it places the field somewhere else, or gives a distance that is not in the VERIFIED FACTS
- "fact_ok": for fact: true if the VERIFIED FACTS support it, false otherwise
Return: {{"statements": [ ... ]}}"""

def judge(client, model, sc, text):
    p = JUDGE.format(names=", ".join(h["name"] for h in sc["racecard"]), passed="; ".join(sc["passed"]), ahead="; ".join(sc["ahead"]),
                     facts="\n".join("- " + f for f in sc["facts"]), text=text)
    for _ in range(2):
        r = client.chat(model=model, messages=[{"role": "user", "content": p}], format="json", options={"temperature": 0, "num_predict": 1500, "num_ctx": 8192})
        try:
            d = json.loads(r["message"]["content"]); st = d.get("statements") if isinstance(d, dict) else d
            if isinstance(st, list): return [s for s in st if isinstance(s, dict)]
        except Exception: pass
    return None

def norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())
COL = [("sky", 10), ("light green", 6), ("green", 6), ("white", 1), ("yellow", 2), ("red", 3), ("black", 4), ("blue", 5), ("brown", 7), ("pink", 8), ("purple", 9), ("violet", 9)]
def split_order(text):
    """-> (ORDER line as list of cap numbers, rest of the text). sky blue caps (10+) are all mapped to 10."""
    m = re.match(r"\s*\**ORDER\**\s*:\s*(.*)", text or "", flags=re.I)
    if not m: return None, text
    line = m.group(1).split("\n")[0]; rest = (text or "")[m.end(1) - len(m.group(1)) + len(line):].strip(); nos = []
    for part in re.split(r"[,;>]| then ", line.lower()):
        for k, n in COL:
            if k in part: nos.append(n); break
    return nos, rest
def order_metrics(sc, nos):
    off = [min(n, 10) for n in sc["order_now"]]
    if not nos: return {"has_order": 0, "leader_ok": 0, "top3_hits": 0, "pairs_ok": None}
    hits = len(set(nos[:3]) & set(off[:3])); seen = [n for n in dict.fromkeys(nos) if n in off]
    pairs = [(a, b) for i, a in enumerate(seen) for b in seen[i+1:]]; ok = sum(off.index(a) < off.index(b) for a, b in pairs)
    return {"has_order": 1, "leader_ok": int(nos[0] == off[0]), "top3_hits": hits, "pairs_ok": round(ok / len(pairs), 2) if pairs else None}
def allowed_numbers(sc):
    return {int(x) for f in sc["facts"] for x in re.findall(r"(\d+) ?(?:metres|metre|m)\b", f)}
def distance_numbers(text):
    return {int(x) for x in re.findall(r"(\d{2,4})[ -]?(?:metres|metre|meters|meter|m)\b", (text or "").lower())}

KEYS = ["true_pos", "false_pos", "named", "loc_ok", "loc_bad", "fact_ok", "fact_bad", "abstain", "vague", "other"]
def verdict(sc, s):
    t = s.get("type"); byname = {norm(h["name"]): h["no"] for h in sc["racecard"]}; order = sc["order_now"]; n = len(order)
    bad_number = bool(distance_numbers(s.get("text")) - allowed_numbers(sc))            # a distance that is not in the verified facts
    if t == "position":
        no = byname.get(norm(s.get("horse")))
        if no is None: return "other"
        if no not in order: return "false_pos"
        true = order.index(no) + 1; checks = []; rk = s.get("rank")
        if isinstance(rk, str) and rk.strip().lower() == "last": checks.append(true >= n - 1)
        elif isinstance(rk, (int, float)) or (isinstance(rk, str) and rk.strip().isdigit()):
            rk = int(rk); checks.append(true == rk if (rk == 1 or true == 1) else abs(true - rk) <= 1)
        other = byname.get(norm(s.get("ahead_of")))
        if other is not None and other in order: checks.append(true < order.index(other) + 1)
        if not checks: return "named"
        return "true_pos" if all(checks) else "false_pos"
    if t == "named": return "named" if norm(s.get("horse")) in byname else "other"
    if t == "location": return "loc_bad" if (bad_number or s.get("loc_ok") is not True) else "loc_ok"
    if t == "fact": return "fact_bad" if (bad_number or s.get("fact_ok") is not True) else "fact_ok"
    if t == "abstain": return "abstain"
    if t == "vague": return "loc_bad" if bad_number else "vague"
    return "loc_bad" if bad_number else "other"

def summarise(rows):
    out = {}
    for c in [c for c in "ARFBCS" if any(r["cond"] == c for r in rows)]:
        rs = [r for r in rows if r["cond"] == c and r.get("verdicts") is not None]; cnt = collections.Counter(v for r in rs for v in r["verdicts"]); k = max(1, len(rs))
        hall = cnt["false_pos"] + cnt["loc_bad"] + cnt["fact_bad"]; good = cnt["true_pos"] + cnt["loc_ok"]; pos = cnt["true_pos"] + cnt["false_pos"]
        out[c] = {"calls": len(rs), "statements": sum(cnt.values()), **{x: cnt[x] for x in KEYS},
                  "HALLUCINATIONS_per_call": round(hall / k, 2), "correct_live_claims_per_call": round(good / k, 2),
                  "verified_facts_per_call": round(cnt["fact_ok"] / k, 2), "vague_per_call": round(cnt["vague"] / k, 2),
                  "position_accuracy": round(cnt["true_pos"] / max(1, pos), 2), "calls_with_any_hallucination": sum(bool({"false_pos", "loc_bad", "fact_bad"} & set(r["verdicts"])) for r in rs)}
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("scenes"); ap.add_argument("responses"); ap.add_argument("out")
    ap.add_argument("--judge", default="qwen3-vl:8b-instruct"); a = ap.parse_args()
    scenes = {s["scene"]: s for s in json.load(open(a.scenes))}
    done = {(r["scene"], r["cond"], r["rep"]): r for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else {}
    import ollama
    rows = []
    with open(a.out, "a") as fh:
        for r in map(json.loads, open(a.responses)):
            key = (r["scene"], r["cond"], r["rep"])
            if key in done: rows.append(done[key]); continue
            nos, body = split_order(r["response"]); r["order_line"] = nos; r["order"] = order_metrics(scenes[r["scene"]], nos)
            st = judge(ollama, a.judge, scenes[r["scene"]], body)
            r["statements"] = st; r["verdicts"] = None if st is None else [verdict(scenes[r["scene"]], s) for s in st]
            fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush(); rows.append(r)
            print(r["scene"], r["cond"], r["verdicts"])
    base = os.path.splitext(a.out)[0]
    with open(base + "_statements.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["scene", "cond", "verdict", "statement", "type", "horse", "rank", "ahead_of", "official_order_now(numbers)", "field_has_passed", "your_check(O/X)"])
        for r in rows:
            for s, v in zip(r.get("statements") or [], r.get("verdicts") or []):
                sc = scenes[r["scene"]]
                w.writerow([r["scene"], r["cond"], v, s.get("text"), s.get("type"), s.get("horse"), s.get("rank"), s.get("ahead_of"), " ".join(map(str, sc["order_now"])), sc["passed"][-1], ""])
    sm = summarise(rows); json.dump(sm, open(base + "_summary.json", "w"), indent=1)
    rowsk = ["calls", "statements"] + KEYS + ["HALLUCINATIONS_per_call", "correct_live_claims_per_call", "verified_facts_per_call", "vague_per_call", "position_accuracy", "calls_with_any_hallucination"]
    print("\n" + f"{'':30}" + "".join(f"{c:>8}" for c in sm))
    for k in rowsk: print(f"{k:30}" + "".join(f"{sm[c][k]:>8}" for c in sm))
    print("\nORDER line (checked by code against the official order, no judge):")
    print(f"{'':30}" + "".join(f"{c:>8}" for c in sm))
    for lab, fn in [("calls_with_ORDER_line", lambda rs: sum(r["order"]["has_order"] for r in rs)),
                    ("leader_correct", lambda rs: sum(r["order"]["leader_ok"] for r in rs)),
                    ("top3_hits_of_3 (mean)", lambda rs: round(sum(r["order"]["top3_hits"] for r in rs) / max(1, len(rs)), 2)),
                    ("pairwise_order_correct (mean)", lambda rs: (lambda v: round(sum(v) / len(v), 2) if v else "-")([r["order"]["pairs_ok"] for r in rs if r["order"]["pairs_ok"] is not None]))]:
        print(f"{lab:30}" + "".join(f"{fn([r for r in rows if r['cond'] == c and r.get('order')]):>8}" for c in sm))
    print("\nfailed to parse:", sum(r.get("verdicts") is None for r in rows), "| statements:", base + "_statements.csv")
