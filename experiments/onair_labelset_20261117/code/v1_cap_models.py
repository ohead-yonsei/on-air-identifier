"""Compare cap-reading models on the 417 human-labelled cap crops. See models/DESIGN_KO.md.

usage: python v1_cap_models.py run MODEL     (calls MODEL on every crop; resumable)
       python v1_cap_models.py score         (scores every model that has been run, plus the stored 9B answers)
"""
import base64
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "models"
ENDPOINT = "http://127.0.0.1:11435"
PROTOCOL = json.loads((ROOT / "code" / "codex_copy" / "cap_classifier.json").read_text())
DEV = {"R01", "R03", "R05", "R07", "R09"}
HARD = (4, 7, 10, 11)
BASE = "qwen3.5:9b"


def riders():
    out = []
    for batch, base in (("1", ROOT), ("2A", ROOT / "batch2"), ("2B", ROOT / "batch2b")):
        for r in json.loads((base / "results" / "evidence.json").read_text()):
            if r["human_number"] and not r["duplicate_of"]:
                out.append({"key": f"{batch}:{r['id']}", "race": r["race"], "human": r["human_number"],
                            "image": base / r["R0_35"]["input"], "q9": r["R0_35"]["vlm"]["number"]})
    return out


def slug(model):
    return model.replace(":", "_").replace("/", "_")


def wait_gpu():
    lock = Path("/home/ohead/icip2027_research_gpu.lock")
    while True:
        held = []
        if lock.exists():
            inode = str(lock.stat().st_ino)
            held = [s for s in Path("/proc/locks").read_text().splitlines() if any(t.endswith(":" + inode) for t in s.split())]
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
        if not held and not [a for a in apps.strip().splitlines() if a and "ollama" not in a.lower()]:
            return
        time.sleep(30)


def run(model):
    tags = json.loads(urllib.request.urlopen(ENDPOINT + "/api/tags", timeout=30).read())
    digest = next(m["digest"] for m in tags["models"] if m["name"] == model)
    M.mkdir(exist_ok=True)
    raw = M / f"raw_{slug(model)}.jsonl"
    prior = [json.loads(x) for x in raw.read_text().splitlines()] if raw.exists() else []
    done = {r["key"] for r in prior if not r["failure"]}
    tries = Counter(r["key"] for r in prior)
    wait_gpu()
    rs = riders()
    for i, r in enumerate(rs):
        while r["key"] not in done and tries[r["key"]] < 2:
            tries[r["key"]] += 1
            img = r["image"].read_bytes()
            msgs = [{"role": "system", "content": PROTOCOL["SYSTEM"]},
                    {"role": "user", "content": PROTOCOL["PROMPT"], "images": [base64.b64encode(img).decode()]}]
            body = json.dumps({"model": model, "messages": msgs, "format": PROTOCOL["SCHEMA"], "stream": False, "think": False,
                               "keep_alive": "10m", "options": PROTOCOL["options"]}).encode()
            rec = {"key": r["key"], "model": model, "digest": digest, "image_sha256": hashlib.sha256(img).hexdigest(),
                   "request_sha256": hashlib.sha256(body).hexdigest(), "attempt": tries[r["key"]]}
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/chat", data=body, headers={"Content-Type": "application/json"}), timeout=600) as resp:
                    text = resp.read().decode()
                cls = json.loads(json.loads(text)["message"]["content"])["color_class"]
                rec.update(failure=False, color_class=cls, number=PROTOCOL["CLASSES"].index(cls))
                done.add(r["key"])
            except Exception as e:
                rec.update(failure=True, error=repr(e))
            rec["ms"] = round((time.perf_counter() - start) * 1000, 1)
            with raw.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        if i == 0:
            mem = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], text=True).strip()
            print(model, "first answer:", rec.get("color_class"), "failure", rec["failure"], "| gpu", mem, flush=True)
            assert not rec["failure"], rec.get("error")
    rows = [json.loads(x) for x in raw.read_text().splitlines()]
    ok = [r["ms"] for r in rows if not r["failure"]]
    print(model, "answers", len(done), "/", len(rs), "failures", sum(r["failure"] for r in rows), "median ms", sorted(ok)[len(ok) // 2] if ok else None)
    try:
        urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/generate", data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=120).read()
    except Exception:
        pass


def score():
    rs = riders()
    answers = {BASE: {r["key"]: r["q9"] for r in rs}}
    for p in sorted(M.glob("raw_*.jsonl")):
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        if rows:
            answers[rows[0]["model"]] = {r["key"]: r["number"] for r in rows if not r["failure"]}
    models = list(answers)

    def tally(sel, model):
        k = Counter()
        for r in sel:
            a = answers[model].get(r["key"])
            k["missing" if a is None else "abstain" if a == 0 else "correct" if a == r["human"] else "wrong"] += 1
        ans = k["correct"] + k["wrong"]
        return {"n": len(sel), "correct": k["correct"], "wrong": k["wrong"], "abstain": k["abstain"], "missing": k["missing"],
                "precision": k["correct"] / ans if ans else None}

    out = {"models": models, "splits": {}}
    for split in ("dev", "test"):
        sel = [r for r in rs if (r["race"] in DEV) == (split == "dev")]
        out["splits"][split] = {m: {"all": tally(sel, m), "hard_4_7_10_11": tally([r for r in sel if r["human"] in HARD], m),
                                    "by_number": {n: tally([r for r in sel if r["human"] == n], m)["correct"] for n in range(1, 12)}} for m in models}
        print(f"\n{split}: {len(sel)} riders | n by number " + " ".join(f"{sum(r['human'] == n for r in sel):>3d}" for n in range(1, 12)))
        for m in models:
            v = out["splits"][split][m]
            a, h = v["all"], v["hard_4_7_10_11"]
            pr = "n/a" if a["precision"] is None else f"{a['precision']:.3f}"
            print(f"  {m:24s} correct {a['correct']:3d} wrong {a['wrong']:3d} abstain {a['abstain']:3d} precision {pr} | hard colours {h['correct']:2d}/{h['n']} (wrong {h['wrong']}) | "
                  + " ".join(f"{v['by_number'][n]:>3d}" for n in range(1, 12)))
    dev = out["splits"]["dev"]
    base_p = dev[BASE]["all"]["precision"]
    ok = [m for m in models if dev[m]["all"]["precision"] is not None and dev[m]["all"]["precision"] >= base_p - 0.03 and dev[m]["all"]["missing"] == 0]
    chosen = max(ok, key=lambda m: (dev[m]["all"]["correct"], -models.index(m)))
    out["chosen_on_dev"] = chosen
    print("\nchosen on dev:", chosen)
    if chosen != BASE:
        sel = [r for r in rs if r["race"] not in DEV]
        imp = sum(answers[chosen].get(r["key"]) == r["human"] and answers[BASE][r["key"]] != r["human"] for r in sel)
        wor = sum(answers[BASE][r["key"]] == r["human"] and answers[chosen].get(r["key"]) != r["human"] for r in sel)
        n, mm = imp + wor, min(imp, wor)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(mm + 1)) / 2 ** n) if n else 1.0
        out["confirm_on_test"] = {"model": chosen, "improved": imp, "worsened": wor, "sign_test_p": p, "better": bool(imp > wor and p < 0.05)}
        print(f"test riders, {BASE} -> {chosen}: improved {imp} / worsened {wor}  p={p:.4f}  better: {out['confirm_on_test']['better']}")
    (M / "score.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    run(sys.argv[2]) if sys.argv[1] == "run" else score()
