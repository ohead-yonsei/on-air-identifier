"""ON-AIR draft benchmark v0: the picture-only baseline with a paid model of the OpenAI API (added 2026-11-21).
usage: python3 baseline_openai.py --name NAME --races 20260905_R09 20260912_R08 ... [--model gpt-6.1-terra] [--prompt free|careful]
                                  [--split test] [--limit N] [--budget 3.0] [--check-model]
       (standard library only; needs the pictures from make_inputs.py in inputs/ and OPENAI_API_KEY in the .env at the repository root)

The same task, pictures, race card and two prompts as baseline_vlm.py (the local model), so the rows can stand in one table.
Items of both tracks (strip and official) of the named races go into one file, baselines/NAME.jsonl; a run can be stopped and
started again. Each line keeps the tokens the API counted and the time the call took.

Money: the run stops before the next call once the spend estimated from the counted tokens passes --budget (US dollars; the prices
per million tokens are arguments because they change). The key is read from .env and is never printed or written anywhere."""
import argparse
import base64
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--model", default="gpt-6.1-terra")
ap.add_argument("--races", nargs="+", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--prompt", default="free", choices=("free", "careful"))
ap.add_argument("--inputs", default=str(HERE / "inputs"))
ap.add_argument("--limit", type=int)
ap.add_argument("--budget", type=float, default=3.0)
ap.add_argument("--price-in", type=float, default=1.0, help="US dollars per million input tokens")
ap.add_argument("--price-out", type=float, default=5.0, help="US dollars per million output tokens")
ap.add_argument("--check-model", action="store_true", help="only ask the API whether the model id exists, then stop")
a = ap.parse_args()

# the texts below are the ones of baseline_vlm.py, unchanged
CAPS = "1번 흰색, 2번 노란색, 3번 빨간색, 4번 검정, 5번 파란색, 6번 초록색, 7번 갈색, 8번 분홍색, 9번 보라색, 10번 하늘색, 11번부터는 하늘색 바탕에 줄무늬"
SYSTEM = {"free": "당신은 한국어 경마 중계 캐스터다. 화면을 보고 지금 누가 앞서 있는지 한 문장으로 중계한다. 말은 이름 대신 번호로 부르고, 번호 뒤에 '번'을 붙인다. "
                  "말 이름과 기수 이름은 말하지 않는다. 한 문장, 45자 안쪽, 캐스터 말투. JSON으로만 답한다.",
          "careful": "당신은 한국어 경마 중계 캐스터다. 화면을 보고 지금 상황을 한 문장으로 중계한다. 말은 이름 대신 번호로 부르고, 번호 뒤에 '번'을 붙인다. "
                     "번호는 화면에서 확실히 알아본 말만 말한다. 확실하지 않으면 번호를 말하지 말고 주어진 경주 정보(경주 번호, 거리, 경과 시간)만으로 중계한다. "
                     "말 이름과 기수 이름은 말하지 않는다. 한 문장, 45자 안쪽, 캐스터 말투. JSON으로만 답한다."}[a.prompt]
SCHEMA = {"type": "object", "properties": {"line": {"type": "string"}}, "required": ["line"], "additionalProperties": False}
API = "https://api.openai.com/v1"


def key():
    env = HERE.parent.parent / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        m = re.match(r"\s*OPENAI_API_KEY\s*=\s*['\"]?([^'\"\s#]+)", line)
        if m:
            return m.group(1)
    raise SystemExit("no OPENAI_API_KEY in the .env at the repository root")


def clean(text):
    """Error texts can repeat part of the key: never keep anything that looks like one."""
    return re.sub(r"sk-[A-Za-z0-9_\-*.]+", "sk-…", text)[:500]


def call(path, body=None, tries=4):
    req = urllib.request.Request(API + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode()), None
        except urllib.error.HTTPError as e:
            text = clean(e.read().decode(errors="replace"))
            if e.code in (429, 500, 502, 503, 504) and k + 1 < tries:
                time.sleep(5 * 2 ** k)
                continue
            return None, f"HTTP {e.code}: {text}"
        except Exception as e:      # noqa: BLE001
            if k + 1 < tries:
                time.sleep(5 * 2 ** k)
                continue
            return None, clean(repr(e))


KEY = key()
if a.check_model:
    res, err = call("/models")
    ids = sorted(m["id"] for m in res["data"]) if res else []
    print("error:", err) if err else print(a.model, "is in the list" if a.model in ids else "is NOT in the list;", "ids like it:", [i for i in ids if "6" in i][:40])
    raise SystemExit

meta = json.loads((HERE / "races.json").read_text())["races"]
items = [json.loads(l) for f in ("items.jsonl", "items_official.jsonl") for l in (HERE / f).read_text().splitlines()]
items = [it for it in items if it["race_key"] in a.races and a.split in ("all", it["split"])]
random.Random(20261121).shuffle(items)          # fixed order, so a run that stops early has covered a random sample
items = items[: a.limit]
out = HERE / "baselines" / f"{a.name}.jsonl"
out.parent.mkdir(exist_ok=True)
old = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
done = {r["id"] for r in old if not r.get("error")}
spent = sum(r.get("usd", 0) for r in old)
print(len(items), "items,", len(done), "done before, spent so far about $%.3f" % spent, flush=True)
n = 0
for it in items:
    if it["id"] in done:
        continue
    if spent >= a.budget:
        print("budget reached, stopping before", it["id"], flush=True)
        break
    r = meta[it["race_key"]]
    user = (f"경주 정보: 서울 {r['race']}경주, {r['distance_m']}m, 출전 번호 {', '.join(map(str, r['runners']))}. 출발 후 {int(it['seconds_since_start'])}초.\n"
            f"번호는 기수 모자색으로 안다: {CAPS}.\n그림 세 장은 1초 전, 0.5초 전, 지금이다. 지금 상황을 중계하라.")
    parts = [{"type": "text", "text": user}]
    for k in (1, 2, 3):
        b64 = base64.b64encode((Path(a.inputs) / f"{it['id']}_{k}.jpg").read_bytes()).decode()
        parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64, "detail": "high"}})
    body = {"model": a.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": parts}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "commentary", "strict": True, "schema": SCHEMA}},
            "max_completion_tokens": 3000}
    start = time.perf_counter()
    res, err = call("/chat/completions", body)
    ms = round((time.perf_counter() - start) * 1000)
    line, usage = "", {}
    if res:
        usage = res.get("usage") or {}
        try:
            line = json.loads(res["choices"][0]["message"]["content"]).get("line", "")
        except Exception as e:      # noqa: BLE001
            err = "unreadable answer: " + clean(repr(e)) + " | " + clean(json.dumps(res.get("choices", [{}])[0], ensure_ascii=False))
    usd = (usage.get("prompt_tokens", 0) * a.price_in + usage.get("completion_tokens", 0) * a.price_out) / 1e6
    spent += usd
    line = re.sub(r"(\d)\s+(번|경주|초|m\b|미터)", r"\1\2", line.strip())
    with out.open("a") as f:
        f.write(json.dumps({"id": it["id"], "text": line, "error": err, "ms": ms, "tokens_in": usage.get("prompt_tokens"),
                            "tokens_out": usage.get("completion_tokens"), "usd": round(usd, 5)}, ensure_ascii=False) + "\n")
    n += 1
    if n <= 3 or n % 20 == 0 or err:
        print(n, it["id"], "->", line or err, f"| {ms} ms, ${usd:.4f}, total ${spent:.3f}", flush=True)
print("done", a.name, n, "new items, spent about $%.3f in all" % spent)
