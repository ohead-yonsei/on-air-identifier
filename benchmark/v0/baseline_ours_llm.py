"""ON-AIR draft benchmark v0: baseline "our flow with a language model writing the sentence".
usage: python3 baseline_ours_llm.py IDENTIFIER_OUTPUT.json [...] --name NAME [--official] [--lang ko|en] [--model qwen3.5:9b] [--split test] [--limit N] [--template-only]
       (standard library only; needs a local Ollama server. No picture is sent: the writer gets text only.)

The flow, per item:
  1. the identifier (as in baseline_ours.py) gives the leading group's numbers and, when it names one, the leader; or nothing
  2. a language model writes one sentence of commentary from those facts and the race card (first attempt = "raw")
  3. guard: the sentence is read with the scorer's own rules (score.claims); it passes only if the numbers it names are exactly the
     numbers given and the leader it names is exactly the leader given (none when none was given), and it uses none of the words for
     claims that were not given (places, overtaking, winning, margins; and, when no leader was given, any word for leading or for an
     order). The word lists are stricter than the scorer on purpose: a sentence the scorer cannot read must not slip through.
     The confirmed information is checked too: a time, race number or distance in the sentence must be the given one, and no other
     figure may appear (in English no number written as a word, so that every figure can be read).
  4. if it does not pass, the model writes once more (other seed, higher temperature); if that fails too, the fixed sentence
     pattern of baseline_ours.py is used
When the identifier gives nothing, the writer is not asked: the fixed confirmed-information sentence is used (source "pattern, nothing given").
Writes baselines/NAME.jsonl (the guarded sentence; "source" = model, model_retry or template; "raw" = the first attempt;
"given" = what the identifier gave) and baselines/NAME_raw.jsonl (the first attempt alone, to score what the writer does unguarded).
Can be stopped and started again."""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("outputs", nargs="+")
ap.add_argument("--name", required=True)
ap.add_argument("--official", action="store_true")
ap.add_argument("--lang", default="ko", choices=("ko", "en"))
ap.add_argument("--model", default="qwen3.5:9b")
ap.add_argument("--endpoint", default="http://127.0.0.1:11435")
ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int)
ap.add_argument("--template-only", action="store_true", help="no model: the fixed sentence pattern in the chosen language")
a = ap.parse_args()

SYSTEM = {
    "ko": "당신은 한국어 경마 중계 캐스터다. 주어진 사실만으로 지금 상황을 한 문장으로 중계한다. 말은 이름 대신 번호로 부르고, 번호 뒤에 '번'을 붙인다. "
          "사실에 주어진 번호만, 주어진 번호는 모두 말한다. 누가 선두인지는 사실에 '선두'가 주어졌을 때만 말하고, 그때는 그 번호가 선두라고 분명히 말한다. "
          "순위(1위·2위), 거리 차이, 추월, 우승, 말 이름, 기수 이름은 말하지 않는다. 경주 번호, 거리, 지난 시간을 말할 때는 주어진 숫자 그대로 아라비아 숫자로 쓴다. 한글과 숫자만 쓴다. 한 문장, 45자 안쪽, 캐스터 말투. JSON으로만 답한다.",
    "en": "You are a horse racing commentator calling a race in English. Call the current moment in one sentence, using only the facts given. "
          "Refer to horses by number, written like 'No. 4', never by name. Write the race as 'Race 3', never 'Race No. 3'. Name every number given in the facts and no other number. "
          "Write each horse separately as 'No. N' ('No. 3 and No. 6', never 'Nos. 3 and 6', never the number as a word). "
          "Say who leads only when the facts give a leader, and then say clearly that this number leads. When the facts give no leader, the horses are 'in the leading group': "
          "never use lead, leads, leader, front, pace, ahead or head for them, because the facts do not say who is in front inside the group. "
          "Do not mention places (first, second), margins, overtaking, winning, horse names or jockey names. "
          "When you give the time, the race number or the distance, use the digits as given ('55 seconds'), never words or minutes. The distance is the length of the race, not where the horses are. "
          "One sentence, at most 20 words, in a race caller's voice. Answer in JSON only."}[a.lang]
SCHEMA = {"type": "object", "properties": {"line": {"type": "string"}}, "required": ["line"], "additionalProperties": False}


def facts_and_rule(r, it, said, lead):
    sec = int(it["seconds_since_start"])
    others = [n for n in said if n != lead]
    if a.lang == "ko":
        f = {"경주": f"서울 {r['race']}경주 {r['distance_m']}m", "출발 뒤 지난 시간": f"{sec}초"}
        if lead is not None:
            f["선두"] = f"{lead}번"
            if others:
                f["선두권의 다른 번호 (선두와의 앞뒤는 모름)"] = [f"{n}번" for n in others]
            rule = f"{lead}번이 선두라고 분명히 말하라." + (" 다른 번호는 선두권에 함께 있다고만 말하라." if others else "")
        elif said:
            f["선두권에 있는 번호 (그 안의 순서는 모름)"] = [f"{n}번" for n in said]
            rule = "주어진 번호가 선두권에 있다고 말하라. 누가 선두인지, 번호들 사이의 순서는 말하지 마라."
        else:
            rule = "지금 누가 앞서는지는 확인되지 않았다. 번호도, 선두나 선두권 이야기도 하지 말고, 경주 정보와 지난 시간만으로 한 문장을 만들어라."
        return "사실: " + json.dumps(f, ensure_ascii=False) + "\n지시: " + rule
    f = {"race": f"Seoul Race {r['race']}, {r['distance_m']} m", "time since the start": f"{sec} seconds"}
    if lead is not None:
        f["leader"] = f"No. {lead}"
        if others:
            f["other numbers in the leading group (order relative to the leader unknown)"] = [f"No. {n}" for n in others]
        rule = f"Say clearly that No. {lead} leads." + (" Say only that the other numbers are in the leading group with it." if others else "")
    elif said:
        f["numbers in the leading group (order among them unknown)"] = [f"No. {n}" for n in said]
        rule = "Say that the given numbers are in the leading group, with the words 'in the leading group'. Do not say who leads (no 'lead', 'leads', 'front'), and do not give an order among them."
    else:
        rule = "It is not confirmed who is in front right now. Name no number and do not speak of a leading group or a leader at all; make one sentence from the race information and the time since the start only."
    return "Facts: " + json.dumps(f) + "\nInstruction: " + rule


def template(r, it, said, lead):
    sec = int(it["seconds_since_start"])
    others = [n for n in said if n != lead]
    if a.lang == "ko":
        if lead is not None:
            return f"선두는 {lead}번" + (", 선두권에 " + ", ".join(f"{n}번" for n in others) + "도 있습니다." if others else "입니다.")
        if said:
            return "선두권에 " + ", ".join(f"{n}번" for n in said) + "이 달리고 있습니다."
        return f"서울 {r['race']}경주 {r['distance_m']}m, 출발 후 {sec}초가 지났습니다."
    join = lambda ns: " and ".join(f"No. {n}" for n in ns) if len(ns) < 3 else ", ".join(f"No. {n}" for n in ns[:-1]) + f" and No. {ns[-1]}"
    if lead is not None:
        return f"No. {lead} leads" + (f", with {join(others)} in the leading group." if others else ".")
    if said:
        return f"{join(said)} {'is' if len(said) == 1 else 'are'} in the leading group."
    return f"Seoul Race {r['race']}, {r['distance_m']} m, {sec} seconds since the start."


def tidy(line):
    line = re.sub(r"\s+", " ", line.strip())
    return re.sub(r"(\d)\s+(번|경주|초|m\b|미터)", r"\1\2", line) if a.lang == "ko" else line


NEVER = {"ko": re.compile(r"\d\s*(?:위|등|착)|[첫둘셋]째|[두세]\s*번째|추월|역전|우승|제치|제쳤|따라잡|마신|차이|순으로|순서|바짝|간발|이름|지점|통과|남기|남은|남았"),
         "en": re.compile(r"\b(?:first|second|third|fourth|1st|2nd|3rd|overtak\w*|pass(?:es|ing|ed)?|wins?|winner|winning|won|lengths?|margin|neck|nose|"
                          r"clear\s+of|ahead\s+of|closing|gaining|challeng\w*|to\s+go|remaining|left|mark)\b|\bat\s+\d[\d,]*\s*(?:m\b|met(?:er|re)s?)", re.I)}[a.lang]
NO_LEADER = {"ko": re.compile(r"선두(?!권)|앞장|앞서|앞선|이끌|맨\s*앞|리드|뒤를|뒤에서|쫓|추격"),
             "en": re.compile(r"\blead(?:s|er|ers)?\b|\bleading\b(?!\s+(?:group|pack|bunch))|\bfront\b(?!\s+(?:group|pack|bunch))|\bpace\b|\bahead\b|\bhead(?:s|ing)?\b|"
                               r"\bbehind\b|\bfollow\w*|\bchas\w+|\btrail\w*", re.I)}[a.lang]


WORDS_EN = re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|fou?rty|fifty|"
                      r"sixty|seventy|eighty|ninety|hundred|thousand|minutes?|half|quarter)\b", re.I)
KINDS = {"ko": (r"(\d+)\s*초", r"(\d+)\s*경주", r"(\d[\d,]*)\s*(?:[mM](?![a-zA-Z])|미터)"),
         "en": (r"(\d+)\s*(?:seconds?|secs?|s\b)", r"\brace\s*(\d+)", r"(\d[\d,]*)\s*(?:m\b|met(?:er|re)s?)")}[a.lang]
SCRIPT = {"ko": re.compile(r"[^\x00-\x7f가-힣·…‘’“”]"), "en": re.compile(r"[^\x00-\x7f‘’“”–—]")}[a.lang]          # other scripts (the model sometimes slips into Chinese)
NOTHING = {"ko": re.compile(r"선두|앞"), "en": re.compile(r"\blead\w*|\bfront\b|\bahead\b", re.I)}[a.lang]       # when nothing was given, the sentence must not speak of a leading group at all
HORSE = {"ko": r"\d+\s*번", "en": r"(?:\bNo\.?|\bnumber|#)\s*\d+"}[a.lang]


def confirmed_wrong(line, r, it):
    """The time, the race number and the distance, when the sentence gives them, must be the given ones; no other figure may appear."""
    want = (int(it["seconds_since_start"]), r["race"], r["distance_m"])
    rest = line
    for pat, w, name in zip(KINDS, want, ("time", "race number", "distance")):
        for m in re.finditer(pat, rest, re.I):
            if int(m.group(1).replace(",", "")) != w:
                return f"wrong {name}"
        rest = re.sub(pat, " ", rest, flags=re.I)
    rest = re.sub(HORSE, " ", rest, flags=re.I)
    if re.search(r"\d", rest):
        return "other figure"
    if a.lang == "en" and WORDS_EN.search(line):
        return "number in words"
    return None


def why_not(line, said, lead, r=None, it=None):
    """None when the sentence passes the guard, else a short reason."""
    if not line:
        return "empty"
    if len(line) > (90 if a.lang == "ko" else 220):
        return "too long"
    if SCRIPT.search(line):
        return "other script"
    if r is not None and confirmed_wrong(line, r, it):
        return confirmed_wrong(line, r, it)
    if not said and lead is None and NOTHING.search(line):
        return "speaks of the lead, nothing given"
    nums, leads = score.claims({"text": line, "lang": a.lang})
    if nums != sorted(said):
        return "extra number" if set(nums) - set(said) else "missing number"
    if leads != ([lead] if lead is not None else []):
        return "leader not said" if not leads else ("leader said, none given" if lead is None else "other leader")
    m = NEVER.search(line) or (lead is None and NO_LEADER.search(line))
    return f"word: {m.group(0)}" if m else None


def passes(line, said, lead, r, it):
    return why_not(line, said, lead, r, it) is None


def ask(user, seed, temperature):
    body = json.dumps({"model": a.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "format": SCHEMA, "stream": False,
                       "think": False, "keep_alive": "10m", "options": {"temperature": temperature, "seed": seed, "num_ctx": 4096, "num_predict": 120, "top_p": 0.9}}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(a.endpoint + "/api/chat", data=body, headers={"Content-Type": "application/json"}), timeout=300) as resp:
            return tidy(json.loads(json.loads(resp.read().decode())["message"]["content"]).get("line", "")), None
    except Exception as e:      # noqa: BLE001
        return "", repr(e)


meta = json.loads((HERE / "races.json").read_text())["races"]
items = {}
for l in (HERE / ("items_official.jsonl" if a.official else "items.jsonl")).read_text().splitlines():
    it = json.loads(l)
    r = meta[it["race_key"]]
    if a.split not in ("all", it["split"]):
        continue
    items[(r["source_folder"], f"R{r['race']:02d}", round(it["video_time"], 3)) if a.official else (r["source_folder"], it["id"].split("_", 1)[1])] = it
todo = []
for path in a.outputs:
    res = json.loads(Path(path).read_text())
    if a.official:
        points = [((res["day"], race, round(p["t"], 3)), p) for race, rr in res["races"].items() for p in rr.get("points", [])]
    else:
        points = [((res["day_dir"], p["frame"]), p) for p in res["points"]]
    todo += [(items[k], p) for k, p in points if k in items]
random.Random(20261119).shuffle(todo)          # fixed order, so a run that is stopped early has covered a random sample
todo = todo[: a.limit]
out, raw_out = HERE / "baselines" / f"{a.name}.jsonl", HERE / "baselines" / f"{a.name}_raw.jsonl"
done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
t0 = n = 0
t0 = time.time()
for k, (it, p) in enumerate(todo):
    if it["id"] in done:
        continue
    r = meta[it["race_key"]]
    said, lead = sorted(p["statements"]), p.get("leader")
    if lead is not None and lead not in said:
        said = sorted(said + [lead])
    if a.template_only or (not said and lead is None):
        # nothing given: the writer is not asked. On the dev split it wrote sentences with a wrong meaning here that the check
        # cannot catch ("the field runs through 1400 meters in just 11 seconds"), and there is nothing for it to add.
        line = template(r, it, said, lead)
        with out.open("a") as f:
            f.write(json.dumps({"id": it["id"], "text": line, "lang": a.lang, "source": "template" if a.template_only else "pattern, nothing given",
                                "given": {"numbers": said, "leader": lead}}, ensure_ascii=False) + "\n")
        if not a.template_only:
            with raw_out.open("a") as f:
                f.write(json.dumps({"id": it["id"], "text": line, "lang": a.lang}, ensure_ascii=False) + "\n")
        n += 1
        continue
    user = facts_and_rule(r, it, said, lead)
    start = time.perf_counter()
    raw, err = ask(user, 20261119 + k, 0.5)
    line, source = raw, "model"
    if not passes(raw, said, lead, r, it):
        line, err2 = ask(user, 20271119 + k, 0.7)
        source = "model_retry"
        if not passes(line, said, lead, r, it):
            line, source = template(r, it, said, lead), "template"
    row = {"id": it["id"], "text": line, "lang": a.lang, "source": source, "raw": raw, "raw_fails": why_not(raw, said, lead, r, it), "given": {"numbers": said, "leader": lead}, "error": err,
           "ms": round((time.perf_counter() - start) * 1000)}
    with out.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with raw_out.open("a") as f:
        f.write(json.dumps({"id": it["id"], "text": raw, "lang": a.lang}, ensure_ascii=False) + "\n")
    n += 1
    if n % 25 == 0:
        print(n, "items,", round((time.time() - t0) / n, 1), "s each;", source, "|", raw, "=>", line, flush=True)
print("done", a.name, n, "new items")
