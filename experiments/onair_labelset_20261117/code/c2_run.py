"""Leader-claim comparison on an unseen day, strip segment and finish stretch. See commentary2/DESIGN_KO.md.

usage: python c2_run.py [--limit N]
Two calls per frame: A (picture only) and B (picture + the identifier's confirmed numbers + rule). The gated condition G is
derived in scoring (B's answer where the identifier confirmed something, nothing otherwise), so it needs no extra call.
Writes commentary2/ledger.jsonl (request before sending) and commentary2/raw.jsonl (response)."""
import argparse
import base64
import hashlib
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "commentary2"
ENDPOINT, MODEL = "http://127.0.0.1:11435", "qwen3.5:9b"
MODEL_DIGEST = "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
OPTIONS = {"temperature": 0, "seed": 20261117, "num_ctx": 16384, "num_predict": 96, "top_k": 20, "top_p": 1, "repeat_penalty": 1}
SCHEMA = {"type": "object", "properties": {"leaders": {"type": "array", "items": {"type": "integer"}, "maxItems": 3}},
          "required": ["leaders"], "additionalProperties": False}
SYSTEM = "You assist a horse-racing commentator. You answer only from what is on screen or what you are told is confirmed."
BASE = ("This is one frame from a live horse-racing broadcast. A horse's number can be told from its jockey's cap colour: "
        "1 white, 2 yellow, 3 red, 4 black, 5 blue, 6 green, 7 brown, 8 pink, 9 purple, 10 sky blue, 11 and above sky blue with stripes. "
        "Question: which horse numbers are in the leading group right now (the front three horses)? List up to three numbers. Return only JSON.")
RULE_B = " Name a number only if you are sure. When you are not sure, state only the confirmed number or numbers."
SETS = [("strip", "otherday4"), ("finish", "finish/otherday4"), ("finish", "finish/otherday3"), ("finish", "finish/otherday2")]


def facts_text(nums):
    if not nums:
        return " A separate identification system could not confirm any number in the leading group for this frame."
    return " Confirmed by a separate identification system: " + ", ".join(f"number {n}" for n in sorted(nums)) + (" is" if len(nums) == 1 else " are") + " in the leading group."


def labels(O, name, past):
    rs = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
    if not rs:
        return []
    P = np.array(json.loads((O / "cnn_probs_two" / f"{name}.json").read_text())["probs"])
    Sc = P.copy()
    q = [r["qwen_number"] for r in rs]
    for i, n in enumerate(q):
        if n:
            Sc[i, n - 1] += 1.0
    ri, cj = linear_sum_assignment(-Sc)
    a = {int(i): int(j) for i, j in zip(ri, cj)}
    return [(rs[i]["box"][3], a[i] + 1, bool(q[i] == a[i] + 1 or P[i, a[i]] >= 0.8)) for i in range(len(rs)) if i in a]


def identifier_facts(O, frame):
    """The identifier as it stands (S0): detection D0, two-day classifier + Qwen, sure rule C2, front two boxes, 2 of the 3 most recent frames."""
    votes = Counter()
    for name, past in ((frame, False), (f"{frame}_m05", True), (f"{frame}_m10", True)):
        for n in {n for _, n, s in sorted(labels(O, name, past), key=lambda x: -x[0])[:2] if s}:
            votes[n] += 1
    return sorted(n for n, v in votes.items() if v >= 2)


def frame_list():
    out = []
    for kind, d in SETS:
        O = ROOT / d
        man = json.loads((O / "manifest.json").read_text())
        order = json.loads((O / "finish_order.json").read_text())["order"] if kind == "finish" else None
        for m in man["frames"]:
            if kind == "finish" and m["race"] not in order:
                continue
            ref = m["strip"][:3] if kind == "strip" else order[m["race"]][:3]
            out.append({"key": f"{d}:{m['frame']}", "kind": kind, "dir": d, "frame": m["frame"], "reference": ref})
    return out


def readlines(p):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def append(p, x):
    with p.open("a") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tags = json.loads(urllib.request.urlopen(ENDPOINT + "/api/tags", timeout=30).read())
    assert next(m["digest"] for m in tags["models"] if m["name"] == MODEL) == MODEL_DIGEST
    frames = frame_list()
    if a.limit:
        frames = frames[:: max(1, len(frames) // a.limit)][: a.limit]
    suffix = "_check" if a.limit else ""
    ledger, raw = C / f"ledger{suffix}.jsonl", C / f"raw{suffix}.jsonl"
    done = {(r["key"], r["condition"]) for r in readlines(raw) if not r["failure"]}
    attempts = Counter((r["key"], r["condition"]) for r in readlines(raw))
    for m in frames:
        O = ROOT / m["dir"]
        bgr = cv2.imread(str(O / "frames" / f"{m['frame']}.png"))
        ok, buf = cv2.imencode(".png", cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA))
        img = base64.b64encode(buf.tobytes()).decode()
        facts = identifier_facts(O, m["frame"])
        for cond in ("A", "B"):
            key = (m["key"], cond)
            while key not in done and attempts[key] < 2:
                attempts[key] += 1
                text = BASE + ("" if cond == "A" else facts_text(facts) + RULE_B) + "\nFRAME: [img]"
                body = json.dumps({"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text, "images": [img]}],
                                   "format": SCHEMA, "stream": False, "think": False, "keep_alive": "10m", "options": OPTIONS}).encode()
                append(ledger, {"key": m["key"], "condition": cond, "attempt": attempts[key], "model": MODEL, "model_digest": MODEL_DIGEST, "system": SYSTEM, "prompt": text,
                                "facts": facts, "image_sha256": hashlib.sha256(buf.tobytes()).hexdigest(), "options": OPTIONS, "request_sha256": hashlib.sha256(body).hexdigest()})
                start = time.perf_counter()
                out = {"key": m["key"], "kind": m["kind"], "condition": cond, "attempt": attempts[key], "facts": facts, "reference": m["reference"]}
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
    print("frames", len(frames), "responses", len(rows), "failures", sum(r["failure"] for r in rows))
    try:
        urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/generate", data=json.dumps({"model": MODEL, "keep_alive": 0}).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=60).read()
    except Exception:
        pass


if __name__ == "__main__":
    main()
