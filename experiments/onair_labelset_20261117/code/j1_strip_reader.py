"""Can Qwen read the broadcaster's running-order strip? Checked against the 68 strips Claude read by eye.

Reads leaders/strips/*.jpg and leaders/strip_order.json. Writes leaders/strip_reader_check.json.
The reader is also importable: read_strip(bgr_crop) -> list of five numbers, or None when no full strip is shown.
"""
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "http://127.0.0.1:11435"
MODEL = "qwen3.5:9b"
SYSTEM = "You read on-screen graphics from a horse-racing broadcast exactly as printed. You never guess."
PROMPT = ("This image is a thin horizontal slice from the bottom of a horse-racing broadcast. Sometimes it contains a running-order bar: "
          "five coloured boxes, each holding a horse number, each followed by that horse's name in capital letters. "
          "If the complete bar with five numbers is visible, set visible to true and list the five numbers from left to right. "
          "If there is no bar, or it is cut off or only partly drawn, set visible to false and return an empty list. "
          "Return only JSON.\nSLICE: [img]")
SCHEMA = {"type": "object", "properties": {"visible": {"type": "boolean"}, "numbers": {"type": "array", "items": {"type": "integer"}}},
          "required": ["visible", "numbers"], "additionalProperties": False}
OPTIONS = {"temperature": 0, "seed": 20261117, "num_ctx": 16384, "num_predict": 96, "top_k": 20, "top_p": 1, "repeat_penalty": 1}


def read_strip(bgr, keep_alive="10m"):
    """Return (numbers or None, raw dict). numbers is a list of five ints in 1..16."""
    ok, buf = cv2.imencode(".png", bgr)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT, "images": [base64.b64encode(buf.tobytes()).decode()]}]
    body = json.dumps({"model": MODEL, "messages": msgs, "format": SCHEMA, "stream": False, "think": False, "keep_alive": keep_alive, "options": OPTIONS}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/chat", data=body, headers={"Content-Type": "application/json"}), timeout=180) as r:
            parsed = json.loads(json.loads(r.read().decode())["message"]["content"])
    except Exception as e:
        return None, {"error": repr(e)}
    nums = parsed.get("numbers") or []
    good = bool(parsed.get("visible")) and len(nums) == 5 and all(isinstance(n, int) and 1 <= n <= 16 for n in nums) and len(set(nums)) == 5
    return (nums if good else None), parsed


if __name__ == "__main__":
    truth = json.loads((ROOT / "leaders" / "strip_order.json").read_text())
    order, none = truth["order"], set(truth["frames_without_strip"])
    rows, start = [], time.perf_counter()
    for p in sorted((ROOT / "leaders" / "strips").glob("*.jpg")):
        f = p.stem
        nums, raw = read_strip(cv2.imread(str(p)))
        want = order.get(f)
        rows.append({"frame": f, "claude": want, "qwen": nums, "raw": raw, "match": nums == want})
    ms = (time.perf_counter() - start) * 1000 / max(1, len(rows))
    with_strip = [r for r in rows if r["claude"]]
    without = [r for r in rows if not r["claude"]]
    out = {"model": MODEL, "frames": len(rows), "ms_per_strip": round(ms, 1),
           "with_strip": {"n": len(with_strip), "exact": sum(r["match"] for r in with_strip), "said_none": sum(r["qwen"] is None for r in with_strip)},
           "without_strip": {"n": len(without), "correctly_none": sum(r["qwen"] is None for r in without)},
           "rows": rows}
    (ROOT / "leaders" / "strip_reader_check.json").write_text(json.dumps(out, indent=1))
    print("frames", len(rows), "| with strip: exact", out["with_strip"]["exact"], "/", out["with_strip"]["n"], "said none", out["with_strip"]["said_none"],
          "| without strip: correctly none", out["without_strip"]["correctly_none"], "/", out["without_strip"]["n"], "|", round(ms), "ms each")
    for r in rows:
        if not r["match"]:
            print("  MISMATCH", r["frame"], "claude", r["claude"], "qwen", r["qwen"], "raw", r["raw"])
    try:
        urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/generate", data=json.dumps({"model": MODEL, "keep_alive": 0}).encode(), headers={"Content-Type": "application/json"}), timeout=60).read()
    except Exception:
        pass
