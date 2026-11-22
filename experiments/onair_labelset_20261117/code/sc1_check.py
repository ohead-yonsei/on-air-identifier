"""score1: look at what rules v0 and v1 read as the leader claim in stored Korean sentences.
usage: sc1_check.py dev|check OUT.json        dev = races whose name hashes even (rules were written looking at these), check = odd
Standard library only. Three kinds of sentences:
  pattern    baselines written by fixed patterns: v1 must read exactly what v0 reads (all races, both halves)
  labelled   first attempts of the local writing model (ours_v11_llm_ko*_raw): the leader that was given to the model is known
  open       picture-only models (local Qwen, Terra): no label; every sentence where v0 and v1 differ is listed, and a fixed sample
             of 40 sentences with a number where neither reads a leader"""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "benchmark" / "v0" / "baselines"
sys.path.insert(0, str(ROOT / "benchmark" / "v0"))
import score

half, out_path = sys.argv[1], Path(sys.argv[2])
want = 0 if half == "dev" else 1
side = lambda i: int(hashlib.md5(i[:12].encode()).hexdigest(), 16) % 2
read = lambda name: [json.loads(l) for l in (B / f"{name}.jsonl").read_text().splitlines() if l]
lead = lambda text, rules: score.claims({"text": text}, rules)[1]
res = {"half": half}

# 1. pattern sentences: identical reading
diff = []
for name in ("confirmed_only", "random_three", "ours_identifier_F2", "ours_identifier_v11", "ours_identifier_v11w"):
    for suf in ("", "_official"):
        for r in read(name + suf):
            if "leader" in r or "numbers" in r:
                continue
            if lead(r["text"], "v0") != lead(r["text"], "v1"):
                diff.append((name + suf, r["id"], r["text"], lead(r["text"], "v0"), lead(r["text"], "v1")))
res["pattern sentences read differently"] = diff

# 2. labelled sentences
given = {r["id"]: r["given"] for suf in ("", "_official") for r in read("ours_v11_llm_ko" + suf)}
rows = [r for suf in ("_raw", "_official_raw") for r in read("ours_v11_llm_ko" + suf) if side(r["id"]) == want and r["id"] in given and r.get("text")]
tab, odd = {"v0": Counter(), "v1": Counter()}, []
for r in rows:
    g = given[r["id"]].get("leader")
    for rules in ("v0", "v1"):
        got = lead(r["text"], rules)
        k = ("given, read right" if got == [g] else "given, not read" if not got else "given, read another") if g is not None else ("none given, none read" if not got else "none given, read one")
        tab[rules][k] += 1
        if rules == "v1" and k in ("given, not read", "given, read another", "none given, read one"):
            odd.append({"id": r["id"], "text": r["text"], "given": g, "v0": lead(r["text"], "v0"), "v1": got, "kind": k})
res["labelled"] = {"sentences": len(rows), "v0": dict(tab["v0"]), "v1": dict(tab["v1"]), "v1 not as given": odd}

# 3. open sentences
openrows = [(name, r) for name in ("vlm_qwen9b_free", "vlm_qwen9b_free_official", "vlm_qwen9b_careful", "vlm_qwen9b_careful_official", "vlm_terra_free_4races", "vlm_terra_careful_4races")
            for r in read(name) if side(r["id"]) == want and r.get("text")]
changed, neither = [], []
cnt = Counter()
for name, r in openrows:
    a, b = lead(r["text"], "v0"), lead(r["text"], "v1")
    cnt["same"] += a == b
    if a != b:
        changed.append({"from": name, "id": r["id"], "text": r["text"], "v0": a, "v1": b})
    elif not a and score.NUMBER.search(r["text"]):
        neither.append({"from": name, "id": r["id"], "text": r["text"]})
random.Random(20261122).shuffle(neither)
res["open"] = {"sentences": len(openrows), "read the same": cnt["same"], "read differently": len(changed), "changed": changed,
               "with a number and no leader read by either": len(neither), "sample of those": neither[:40]}
out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("pattern sentences read differently:", len(diff))
for d in diff[:10]:
    print("   ", d)
print("labelled:", len(rows), "\n   v0", dict(tab["v0"]), "\n   v1", dict(tab["v1"]))
for o in odd:
    print("   ", o["kind"], "|", o["given"], o["v0"], o["v1"], "|", o["text"])
print("open:", len(openrows), "same", cnt["same"], "changed", len(changed), "| number but no leader by either:", len(neither))
kinds = Counter((bool(c["v0"]), bool(c["v1"])) for c in changed)
print("   changed by kind (v0 read, v1 read):", dict(kinds))
for c in changed:
    print("   ", c["from"][4:16], c["v0"], "->", c["v1"], "|", c["text"])
