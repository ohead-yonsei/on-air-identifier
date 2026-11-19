"""bench3: look at what the writer did (for prompt work on the dev split, and for the hand check on the test split).
usage: bx1_devlook.py BASELINE.jsonl [...] [--sample N] [--seed S] [--only model|model_retry|template|failed|written]        (standard library only)
Prints where the checked sentence came from, why first attempts failed the check, and a random sample of lines."""
import argparse
import collections
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--sample", type=int, default=30)
ap.add_argument("--seed", type=int, default=20261119)
ap.add_argument("--only")
a = ap.parse_args()
rows = [json.loads(l) for f in a.files for l in open(f)]
n = len(rows)
src = collections.Counter(r["source"] for r in rows)
for r in rows:
    r.setdefault("raw", r["text"])
why = collections.Counter(r["raw_fails"] for r in rows if r.get("raw_fails"))
print(n, "items;", ", ".join(f"{k} {v} ({100 * v / n:.1f}%)" for k, v in src.most_common()))
kinds = collections.Counter("leader given" if r["given"]["leader"] is not None else ("numbers only" if r["given"]["numbers"] else "nothing") for r in rows)
for k in kinds:
    sub = [r for r in rows if ("leader given" if r["given"]["leader"] is not None else ("numbers only" if r["given"]["numbers"] else "nothing")) == k]
    print(f"  {k}: {len(sub)} items, first attempt passes {100 * sum(not r.get('raw_fails') for r in sub) / len(sub):.1f}%, pattern used {100 * sum(r['source'] == 'template' for r in sub) / len(sub):.1f}%")
print("first attempt failed because:", ", ".join(f"{k} {v}" for k, v in why.most_common(25)))
print("ms per item (median):", sorted(r.get("ms", 0) for r in rows)[n // 2])
pick = [r for r in rows if not a.only or (r.get("raw_fails") if a.only == "failed" else r["source"] in ("model", "model_retry") if a.only == "written" else r["source"] == a.only)]
random.Random(a.seed).shuffle(pick)
for r in pick[: a.sample]:
    g = r["given"]
    print(f"{r['id']} | given {g['numbers']} leader {g['leader']} | {r['source']}{' [' + r['raw_fails'] + ']' if r.get('raw_fails') else ''}\n    raw: {r['raw']}\n    out: {r['text']}")
