"""ON-AIR draft benchmark v0: baseline "a vision-language model looks at the pictures and commentates".
usage: python3 baseline_vlm.py --name NAME [--model qwen3.5:9b] [--split test] [--prompt free|careful] [--limit N]
       (standard library only; needs a local Ollama server and the pictures from make_inputs.py)

The model gets the three pictures of the item (strip hidden) and the race card, and writes one sentence of commentary.
Two standard prompts:
  free     commentate what is happening now (who is in front)
  careful  the same, plus: say a number only when sure; otherwise talk about confirmed information only
Writes baselines/NAME.jsonl; can be stopped and started again (finished items are kept).

The prompts contain no example number on purpose: a first version said "write numbers like '4번'", and the local model then
answered "4번이 앞서고 있습니다" for 290 of 300 items (kept in baselines/discarded/)."""
import argparse
import base64
import json
import random
import re
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--model", default="qwen3.5:9b")
ap.add_argument("--endpoint", default="http://127.0.0.1:11435")
ap.add_argument("--split", default="test")
ap.add_argument("--prompt", default="free", choices=("free", "careful"))
ap.add_argument("--inputs", default=str(HERE / "inputs"))
ap.add_argument("--limit", type=int)
a = ap.parse_args()

CAPS = "1번 흰색, 2번 노란색, 3번 빨간색, 4번 검정, 5번 파란색, 6번 초록색, 7번 갈색, 8번 분홍색, 9번 보라색, 10번 하늘색, 11번부터는 하늘색 바탕에 줄무늬"
SYSTEM = {"free": "당신은 한국어 경마 중계 캐스터다. 화면을 보고 지금 누가 앞서 있는지 한 문장으로 중계한다. 말은 이름 대신 번호로 부르고, 번호 뒤에 '번'을 붙인다. "
                  "말 이름과 기수 이름은 말하지 않는다. 한 문장, 45자 안쪽, 캐스터 말투. JSON으로만 답한다.",
          "careful": "당신은 한국어 경마 중계 캐스터다. 화면을 보고 지금 상황을 한 문장으로 중계한다. 말은 이름 대신 번호로 부르고, 번호 뒤에 '번'을 붙인다. "
                     "번호는 화면에서 확실히 알아본 말만 말한다. 확실하지 않으면 번호를 말하지 말고 주어진 경주 정보(경주 번호, 거리, 경과 시간)만으로 중계한다. "
                     "말 이름과 기수 이름은 말하지 않는다. 한 문장, 45자 안쪽, 캐스터 말투. JSON으로만 답한다."}[a.prompt]
SCHEMA = {"type": "object", "properties": {"line": {"type": "string"}}, "required": ["line"], "additionalProperties": False}
OPTIONS = {"temperature": 0, "seed": 20261118, "num_ctx": 16384, "num_predict": 120, "top_p": 1}

meta = json.loads((HERE / "races.json").read_text())["races"]
items = [json.loads(l) for l in (HERE / "items.jsonl").read_text().splitlines()]
items = [it for it in items if a.split in ("all", it["split"])]
random.Random(20261118).shuffle(items)          # fixed order, so a run that is stopped early has covered a random sample
items = items[: a.limit]
out = HERE / "baselines" / f"{a.name}.jsonl"
out.parent.mkdir(exist_ok=True)
done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
t0, n = time.time(), 0
for it in items:
    if it["id"] in done:
        continue
    r = meta[it["race_key"]]
    user = (f"경주 정보: 서울 {r['race']}경주, {r['distance_m']}m, 출전 번호 {', '.join(map(str, r['runners']))}. 출발 후 {int(it['seconds_since_start'])}초.\n"
            f"번호는 기수 모자색으로 안다: {CAPS}.\n그림 세 장은 1초 전, 0.5초 전, 지금이다. 지금 상황을 중계하라.")
    imgs = [base64.b64encode((Path(a.inputs) / f"{it['id']}_{k}.jpg").read_bytes()).decode() for k in (1, 2, 3)]
    body = json.dumps({"model": a.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user, "images": imgs}],
                       "format": SCHEMA, "stream": False, "think": False, "keep_alive": "10m", "options": OPTIONS}).encode()
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(a.endpoint + "/api/chat", data=body, headers={"Content-Type": "application/json"}), timeout=300) as resp:
            line, err = json.loads(json.loads(resp.read().decode())["message"]["content"]).get("line", ""), None
    except Exception as e:      # noqa: BLE001
        line, err = "", repr(e)
    line = re.sub(r"(\d)\s+(번|경주|초|m\b|미터)", r"\1\2", line.strip())
    with out.open("a") as f:
        f.write(json.dumps({"id": it["id"], "text": line, "error": err, "ms": round((time.perf_counter() - start) * 1000)}, ensure_ascii=False) + "\n")
    n += 1
    if n % 25 == 0:
        print(n, "items,", round((time.time() - t0) / n, 1), "s each;", it["id"], "->", line, flush=True)
print("done", a.name, n, "new items")
