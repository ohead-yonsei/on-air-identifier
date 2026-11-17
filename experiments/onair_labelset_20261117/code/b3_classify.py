"""Batch 2B copy of s3_classify.py (same logic, outputs under batch2b/).

Stage 3: per-rider cap evidence for every condition in DESIGN_KO.md.

For each rider box: R0_27 / R0_35 cap regions, the v4 HSV rule scores (Codex code, copied
unchanged), the 21 colour features for the learned classifier, and one Qwen call on R0_35.
Writes results/evidence.json, raw/request_ledger.jsonl, raw/vlm.jsonl, inputs/*.png.
"""
import base64
import hashlib
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "batch2b"   # outputs and labels of the 2B run; code and the Codex copies stay in ROOT
FRAMES = ROOT / "batch2" / "frames"   # 2B frames were extracted together with 2A
SRC = Path("/home/ohead/ai-workspace/projects/onair_overnight_20261117")
sys.path.insert(0, str(ROOT / "code" / "codex_copy"))
import v4_color as v4  # noqa: E402  (unchanged copy of the Codex file)

ENDPOINT = "http://127.0.0.1:11435"
MODEL = "qwen3.5:9b"
MODEL_DIGEST = "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
MAX_VLM = 300
PROTOCOL = json.loads((ROOT / "code" / "codex_copy" / "cap_classifier.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def append(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")


def readlines(p):
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []


def clip(b, w, h):
    l, t, r, d = b
    b = [max(0, math.floor(l)), max(0, math.floor(t)), min(w, math.ceil(r)), min(h, math.ceil(d))]
    return b if (b[2] - b[0]) * (b[3] - b[1]) > 0 else None


def region(b, method, w, h):
    l, t, r, d = b
    bw, bh = r - l, d - t
    if method == "R0_27":
        return clip([l + .08 * bw, t, r - .08 * bw, t + max(4, .27 * bh)], w, h)
    return clip([l, t, r, t + .35 * bh], w, h)


def background_lab(im, person):
    l, t, r, d = clip(person, im.shape[1], im.shape[0])
    lab = cv2.cvtColor(im[t:d, l:r], cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.median(np.concatenate([lab[0], lab[:, 0], lab[:, -1]]), axis=0)


def hsv_scores(im, box, person, width):
    """Same computation as hsv_classify() in the Codex classify_regions.py."""
    bg = background_lab(im, person)
    l, t, r, d = box
    cap = im[t:d, l:r]
    fg = (np.linalg.norm(cv2.cvtColor(cap, cv2.COLOR_BGR2LAB).astype(np.float32) - bg, axis=2) > 20).reshape(-1).astype(float)
    ch, known = v4.hist(v4.quant(np.ascontiguousarray(cap)), fg + 1e-9)
    scores = [v4.cap_score(ch, n) * min(1., known * 4) for n in range(1, 12)]
    rank = np.argsort(scores)[::-1]
    best, margin = float(scores[rank[0]]), float(scores[rank[0]] - scores[rank[1]])
    unknown = best < .25 or margin < .08
    return {"number": 0 if unknown else int(rank[0]) + 1, "top_candidate": int(rank[0]) + 1,
            "all_scores": [float(s) for s in scores], "known_fraction": float(known), "histogram": ch.tolist()}


def features(im, box, person):
    """21 fixed colour features of the foreground pixels of a cap region (see DESIGN_KO.md)."""
    bg = background_lab(im, person)
    l, t, r, d = box
    cap = im[t:d, l:r]
    lab = cv2.cvtColor(cap, cv2.COLOR_BGR2LAB).astype(np.float32)
    fg = np.linalg.norm(lab - bg, axis=2) > 20
    if int(fg.sum()) < 16:
        return None
    hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0][fg], hsv[..., 1][fg], hsv[..., 2][fg]
    wgt = (s / 255) * (v / 255)
    hh = np.bincount(np.minimum((h / 15).astype(int), 11), weights=wgt, minlength=12).astype(float)
    hh = hh / max(hh.sum(), 1e-6)
    extra = [float(((s < 40) & (v > 200)).mean()), float((v < 70).mean()),
             float(s.mean() / 255), float(s.std() / 255), float(v.mean() / 255), float(v.std() / 255)]
    return [float(x) for x in hh] + extra + [float(x) / 255 for x in lab[fg].mean(axis=0)]


def wait_gpu():
    lock = Path("/home/ohead/icip2027_research_gpu.lock")
    while True:
        held = []
        if lock.exists():
            inode = str(lock.stat().st_ino)
            held = [s for s in Path("/proc/locks").read_text().splitlines() if any(t.endswith(":" + inode) for t in s.split())]
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
        apps = [a for a in apps.strip().splitlines() if a and "ollama" not in a.lower()]
        if not held and not apps:
            return
        print("gpu busy, waiting", flush=True)
        time.sleep(30)


def vlm(item, image, last):
    prior = [r for r in readlines(BASE / "raw" / "vlm.jsonl") if r["item_id"] == item]
    if prior and not prior[-1]["failure"]:
        return prior[-1]
    rec = prior[-1] if prior else None
    for attempt in range(len(prior) + 1, 3):
        ledger = BASE / "raw" / "request_ledger.jsonl"
        assert len(readlines(ledger)) < MAX_VLM, "VLM cap reached"
        msgs = [{"role": "system", "content": PROTOCOL["SYSTEM"]},
                {"role": "user", "content": PROTOCOL["PROMPT"], "images": [base64.b64encode(image.read_bytes()).decode()]}]
        payload = {"model": MODEL, "messages": msgs, "format": PROTOCOL["SCHEMA"], "stream": False, "think": False,
                   "keep_alive": 0 if last else "10m", "options": PROTOCOL["options"]}
        body = json.dumps(payload).encode()
        rec = {"item_id": item, "attempt": attempt, "model": MODEL, "model_digest": MODEL_DIGEST,
               "image": str(image.relative_to(BASE)), "image_sha256": sha(image),
               "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
               "schema": PROTOCOL["SCHEMA"], "options": PROTOCOL["options"], "request_sha256": hashlib.sha256(body).hexdigest()}
        append(ledger, rec)
        raw, start = None, time.perf_counter()
        try:
            req = urllib.request.Request(ENDPOINT + "/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = r.read().decode()
            parsed = json.loads(json.loads(raw)["message"]["content"])
            assert set(parsed) == {"color_class"} and parsed["color_class"] in PROTOCOL["CLASSES"]
            rec.update(failure=False, parsed=parsed, number=PROTOCOL["CLASSES"].index(parsed["color_class"]))
        except Exception as e:  # recorded, not hidden
            rec.update(failure=True, error=repr(e))
        rec.update(raw_http_response=raw, http_elapsed_ms=(time.perf_counter() - start) * 1000)
        append(BASE / "raw" / "vlm.jsonl", rec)
        if not rec["failure"]:
            return rec
    return rec


def main():
    tags = json.loads(urllib.request.urlopen(ENDPOINT + "/api/tags", timeout=30).read())
    digest = next(m["digest"] for m in tags["models"] if m["name"] == MODEL)
    assert digest == MODEL_DIGEST, digest
    labels = json.loads((BASE / "labels" / "human_labels.json").read_text())["riders"]
    (BASE / "inputs").mkdir(exist_ok=True)
    (BASE / "results").mkdir(exist_ok=True)
    rows, todo, frame_cache = [], [], {}
    for r in labels:
        if r["frame"] not in frame_cache:
            frame_cache[r["frame"]] = cv2.imread(str(FRAMES / f"{r['frame']}.png"))
        im = frame_cache[r["frame"]]
        H, W = im.shape[:2]
        row = {k: r[k] for k in ("id", "race", "frame", "human_number", "duplicate_of", "height_px", "person_box")}
        for method in ("R0_27", "R0_35"):
            box = region(r["person_box"], method, W, H)
            row[method] = {"box": box, "hsv": hsv_scores(im, box, r["person_box"], W), "features": features(im, box, r["person_box"])}
        box = row["R0_35"]["box"]
        crop = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).crop(box)
        can = Image.new("RGB", (224, 224), (96, 96, 96))
        patch = ImageOps.contain(crop, (224, 224), Image.Resampling.LANCZOS)
        can.paste(patch, ((224 - patch.width) // 2, (224 - patch.height) // 2))
        ip = BASE / "inputs" / f"{r['id']}_R0_35.png"
        can.save(ip)
        row["R0_35"]["input"] = str(ip.relative_to(BASE))
        row["R0_35"]["native_size"] = list(crop.size)
        rows.append(row)
        if r["human_number"] and not r["duplicate_of"]:
            todo.append((row, ip))
    out = BASE / "results" / "evidence.json"
    out.write_text(json.dumps(rows))
    print("rows", len(rows), "vlm items", len(todo), flush=True)
    wait_gpu()
    for i, (row, ip) in enumerate(todo):
        rec = vlm(row["id"] + "_R0_35", ip, last=(i == len(todo) - 1))
        row["R0_35"]["vlm"] = {"number": rec.get("number", 0), "failure": rec["failure"], "http_elapsed_ms": rec["http_elapsed_ms"]}
        if i == 0:
            print("first call:", row["id"], rec.get("parsed"), "failure", rec["failure"], flush=True)
            assert not rec["failure"], rec.get("error")
        if (i + 1) % 25 == 0:
            out.write_text(json.dumps(rows))
            print("vlm", i + 1, "/", len(todo), flush=True)
    out.write_text(json.dumps(rows))
    v = readlines(BASE / "raw" / "vlm.jsonl")
    print("done. vlm requests", len(readlines(BASE / "raw" / "request_ledger.jsonl")), "failures", sum(x["failure"] for x in v))


if __name__ == "__main__":
    main()
