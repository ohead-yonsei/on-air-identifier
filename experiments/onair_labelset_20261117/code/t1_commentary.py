"""Leader-claim comparison with and without identifier facts. See commentary/DESIGN_KO.md.

usage: python t1_commentary.py test            (all four conditions on the held-out races)
       python t1_commentary.py dev --limit 5   (format check only)
Writes commentary/ledger_<split>.jsonl (request before sending) and commentary/raw_<split>.jsonl (response).
"""
import argparse
import base64
import hashlib
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "stripset"
C = ROOT / "commentary"
ENDPOINT, MODEL = "http://127.0.0.1:11435", "qwen3.5:9b"
MODEL_DIGEST = "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
OPTIONS = {"temperature": 0, "seed": 20261117, "num_ctx": 16384, "num_predict": 96, "top_k": 20, "top_p": 1, "repeat_penalty": 1}
SCHEMA = {"type": "object", "properties": {"leaders": {"type": "array", "items": {"type": "integer"}, "maxItems": 3}},
          "required": ["leaders"], "additionalProperties": False}
SYSTEM = "You assist a horse-racing commentator. You answer only from what is on screen or what you are told is confirmed."
BASE = ("This is one frame from a live horse-racing broadcast. A horse's number can be told from its jockey's cap colour: "
        "1 white, 2 yellow, 3 red, 4 black, 5 blue, 6 green, 7 brown, 8 pink, 9 purple, 10 sky blue, 11 and above sky blue with stripes. "
        "Question: which horse numbers are in the leading group right now (the front three horses)? List up to three numbers. Return only JSON.")
RULE = " Name a number only if you are sure. If you are not sure of any, return an empty list."
RULE_B = " Name a number only if you are sure. When you are not sure, state only the confirmed number or numbers."


def facts_text(nums):
    if not nums:
        return " A separate identification system could not confirm any number in the leading group for this frame."
    return " Confirmed by a separate identification system: " + ", ".join(f"number {n}" for n in sorted(nums)) + (" is" if len(nums) == 1 else " are") + " in the leading group."


def identifier_facts(frame):
    """multi/ rule: sure among the front two boxes in at least 2 of the 3 most recent frames."""
    votes = Counter()
    for p in (S / "out" / "D0" / f"{frame}.json", S / "out_past" / "D0" / f"{frame}_m05.json", S / "out_past" / "D0" / f"{frame}_m10.json"):
        riders = json.loads(p.read_text())["riders"]
        front = sorted(riders, key=lambda r: -r["box"][3])[:2]
        for n in {r["no"] for r in front if r["state"] == "sure"}:
            votes[n] += 1
    return sorted(n for n, v in votes.items() if v >= 2)


def prompt_for(cond, facts):
    return BASE + {"A": "", "R": RULE, "F": facts_text(facts), "B": facts_text(facts) + RULE_B}[cond] + "\nFRAME: [img]"


def readlines(p):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def append(p, x):
    with p.open("a") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tags = json.loads(urllib.request.urlopen(ENDPOINT + "/api/tags", timeout=30).read())
    assert next(m["digest"] for m in tags["models"] if m["name"] == MODEL) == MODEL_DIGEST
    frames = [m for m in json.loads((S / "manifest.json").read_text())["frames"] if m["split"] == a.split]
    if a.limit:
        frames = frames[:: max(1, len(frames) // a.limit)][: a.limit]
    C.mkdir(exist_ok=True)
    ledger, raw = C / f"ledger_{a.split}.jsonl", C / f"raw_{a.split}.jsonl"
    done = {(r["frame"], r["condition"]) for r in readlines(raw) if not r["failure"]}
    attempts = Counter((r["frame"], r["condition"]) for r in readlines(raw))
    for m in frames:
        bgr = cv2.imread(str(S / "frames" / f"{m['frame']}.png"))
        small = cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", small)
        img = base64.b64encode(buf.tobytes()).decode()
        facts = identifier_facts(m["frame"])
        for cond in ("A", "R", "F", "B"):
            key = (m["frame"], cond)
            while key not in done and attempts[key] < 2:
                attempts[key] += 1
                text = prompt_for(cond, facts)
                body = json.dumps({"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text, "images": [img]}],
                                   "format": SCHEMA, "stream": False, "think": False, "keep_alive": "10m", "options": OPTIONS}).encode()
                rec = {"frame": m["frame"], "condition": cond, "attempt": attempts[key], "model": MODEL, "model_digest": MODEL_DIGEST, "system": SYSTEM, "prompt": text,
                       "facts": facts, "image_sha256": hashlib.sha256(buf.tobytes()).hexdigest(), "options": OPTIONS, "request_sha256": hashlib.sha256(body).hexdigest()}
                append(ledger, rec)
                start, out = time.perf_counter(), {"frame": m["frame"], "condition": cond, "attempt": attempts[key], "facts": facts, "strip": m["strip"]}
                try:
                    with urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/chat", data=body, headers={"Content-Type": "application/json"}), timeout=300) as r:
                        rawtext = r.read().decode()
                    leaders = json.loads(json.loads(rawtext)["message"]["content"])["leaders"]
                    assert isinstance(leaders, list) and len(leaders) <= 3 and all(isinstance(n, int) for n in leaders)
                    out.update(failure=False, leaders=leaders, raw=rawtext)
                    done.add(key)
                except Exception as e:
                    out.update(failure=True, error=repr(e))
                out["ms"] = round((time.perf_counter() - start) * 1000, 1)
                append(raw, out)
    rows = readlines(raw)
    print(a.split, "frames", len(frames), "responses", len(rows), "failures", sum(r["failure"] for r in rows))
    try:
        urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/generate", data=json.dumps({"model": MODEL, "keep_alive": 0}).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=60).read()
    except Exception:
        pass


if __name__ == "__main__":
    main()
