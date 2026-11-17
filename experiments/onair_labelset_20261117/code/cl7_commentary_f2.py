"""AI commentary demo with order (run 3): like run 2 of cl2_commentary.py, but the identifier's leader and order are also given.
usage: cl5_commentary_order.py DAY [RACE ...]     reads live_commentary/DAY/results_f2s/R??.json (identify_l1.py with leader and order)
and race_facts.json; writes live_commentary/DAY/commentary_v4/. Text only: no picture is sent, and no picture-only sentence is made
(run 2 already has those). The commentary system never sees the running-order strip. See night/DESIGN_KO.md.

In leader mode (same rule as run 2):
  - if the identifier gives a leader that is among the stated numbers, the sentence must name that number as the leader
    (LEADER_CLAIM must find exactly that number) and may say the others follow it only when the identifier's order says the
    leader is ahead of every other stated number; otherwise words about following are not allowed
  - if the identifier gives no leader, no leader may be claimed and no order may be said (as in run 2)"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
day, only = sys.argv[1], set(sys.argv[2:])
O = ROOT / "live_commentary" / day
FACTS = json.loads((O / "race_facts.json").read_text())
ENDPOINT, MODEL = "http://127.0.0.1:11435", "qwen3.5:9b"
CAP = {1: "흰색", 2: "노란색", 3: "빨간색", 4: "검정", 5: "파란색", 6: "초록색", 7: "갈색", 8: "분홍색", 9: "보라색", 10: "하늘색", 11: "하늘색 줄무늬"}
PAST = ("조금 전", "앞서", "직전까지", "방금 전")
BANNED = re.compile(r"\d\s*위|우승|1등|역전|이름")
LEADER_CLAIM = re.compile(r"선두는\s*(?:[가-힣 ]{0,12}모자\s*)?(\d+)번|선두(?:에|로)\s*(?:[가-힣 ]{0,12}모자\s*)?(\d+)번|(\d+)번이\s*선두(?!권)|(\d+)번이\s*(?:맨\s*)?앞(?:장|서)|(\d+)번이\s*(?:경주를\s*)?이끌")
ANY_LEADER_WORD = re.compile(r"선두(?!권)|맨\s*앞|앞장|이끌")
FOLLOW = re.compile(r"뒤|추격|쫓|따르|따라")
LEDGER, RAW = O / "ledger_v4.jsonl", O / "raw_v4.jsonl"


def expected_seconds(d):
    return d * (0.062 if d <= 1200 else 0.0635 if d <= 1400 else 0.066)


def ask(messages, schema, seed, temperature, tag=""):
    body = json.dumps({"model": MODEL, "messages": messages, "format": schema, "stream": False, "think": False, "keep_alive": "10m",
                       "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192, "num_predict": 160, "top_p": 0.9}}).encode()
    with LEDGER.open("a") as f:
        f.write(json.dumps({"tag": tag, "seed": seed, "temperature": temperature, "messages": messages}, ensure_ascii=False) + "\n")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/chat", data=body, headers={"Content-Type": "application/json"}), timeout=300) as r:
            out = json.loads(json.loads(r.read().decode())["message"]["content"])
        err = None
    except Exception as e:
        out, err = None, repr(e)
    with RAW.open("a") as f:
        f.write(json.dumps({"tag": tag, "out": out, "error": err, "ms": round((time.perf_counter() - start) * 1000)}, ensure_ascii=False) + "\n")
    return out


def tidy(line):
    return re.sub(r"(\d)\s+(번|경주|초|m\b|미터)", r"\1\2", line.strip())


def numbers_in(line):
    return sorted({int(n) for n in re.findall(r"(\d+)\s*번", line)})


def leader_claims(line):
    return sorted({int(g) for m in LEADER_CLAIM.finditer(line) for g in m.groups() if g})


SYSTEM = ("당신은 한국어 경마 중계 캐스터다. 주어진 사실만으로 중계한다. 말 이름, 기수 이름, 순위(1위·2위), 거리 차이, 우승은 말하지 않는다. "
          "선두가 누구인지는 사실에 '지금 선두'가 주어졌을 때만 말한다. 번호는 '4번'처럼 쓰고, 사실에 주어진 번호만 쓴다. 한 문장, 45자 안쪽, 캐스터 말투로 쓴다. JSON으로만 답한다.")
LINE = {"type": "object", "properties": {"line": {"type": "string"}}, "required": ["line"], "additionalProperties": False}


def check(line, mode, allowed, recall, leader, follow_ok):
    if not line or len(line) > 80 or BANNED.search(line):
        return False
    nums = numbers_in(line)
    if mode == "leader":
        if not nums or not set(nums) <= set(allowed):
            return False
        if leader is None:
            return not ANY_LEADER_WORD.search(line) and not FOLLOW.search(line)
        return leader_claims(line) == [leader] and (follow_ok or not FOLLOW.search(line))
    return not ANY_LEADER_WORD.search(line) and set(nums) <= set(recall) and (not nums or any(w in line for w in PAST))


def template(mode, allowed, recall, phase, info, leader, follow_ok):
    if mode == "leader":
        if leader is None:
            return "선두권에는 " + ", ".join(f"{CAP[n]} 모자 {n}번" for n in allowed) + "이 달리고 있습니다."
        rest = [n for n in allowed if n != leader]
        if not rest:
            return f"선두는 {CAP[leader]} 모자 {leader}번입니다."
        tail = ", ".join(f"{n}번" for n in rest)
        return f"선두는 {CAP[leader]} 모자 {leader}번, " + (f"그 뒤를 {tail}이 따르고 있습니다." if follow_ok else f"선두권에는 {tail}도 달리고 있습니다.")
    if recall:
        return "조금 전까지 선두권은 " + ", ".join(f"{n}번" for n in recall) + "이었습니다."
    return f"{info['place']} {info['race']}경주 {info['distance']}m, {phase}입니다."


summary = []
for res_path in sorted((O / "results_f2s").glob("R??.json")):
    race = res_path.stem
    if only and race not in only:
        continue
    res = json.loads(res_path.read_text())
    distance, start_t = FACTS[race]["distance"], FACTS[race]["start_video_time"]
    info = {"place": "서울", "race": int(race[1:]), "distance": distance, "start_video_time": start_t, "facts_source": "race_facts.json (checked by eye)"}
    T = expected_seconds(distance)
    rows, prev_lines, last_leader = [], [], None
    for k, p in enumerate(res["points"]):
        elapsed = p["t"] - start_t
        if not (0 <= elapsed <= T + 3):
            continue
        phase = "출발 직후" if elapsed < 8 else "경주 중반" if elapsed <= T - 30 else "경주 후반" if elapsed <= T - 8 else "결승선 직전" if elapsed <= T + 1 else "결승선 통과 무렵"
        mode = "leader" if p["statements"] and 8 <= elapsed <= T - 30 else "confirmed"
        allowed = p["statements"] if mode == "leader" else []
        leader = p.get("leader") if mode == "leader" and p.get("leader") in allowed else None
        ahead = {(o["ahead"], o["behind"]) for o in p.get("order", [])}
        others = [n for n in allowed if n != leader]
        follow_ok = bool(leader is not None and others and all((leader, n) in ahead for n in others))
        recall = last_leader["numbers"] if mode == "confirmed" and last_leader and p["t"] - last_leader["t"] <= 9 else []
        facts = {"경주": f"서울 {info['race']}경주 {distance}m", "출발 뒤 지난 시간": f"{elapsed:.0f}초", "구간": phase}
        left = round(distance * (1 - elapsed / T) / 100) * 100
        if 100 <= left < distance and elapsed >= 8:
            facts["남은 거리(시계로 어림)"] = f"약 {left}m"
        if mode == "leader":
            if leader is not None:
                facts["지금 선두"] = f"{leader}번({CAP[leader]} 모자)"
                if others:
                    facts["선두권의 다른 번호" + (" (선두 뒤에 있음)" if follow_ok else " (선두와의 앞뒤는 확인되지 않음)")] = [f"{n}번({CAP[n]} 모자)" for n in others]
                rule = (f"'선두는 {leader}번'처럼 {leader}번이 선두라고 분명히 말하라. "
                        + ("다른 번호는 선두를 뒤따른다고 말해도 된다." if follow_ok else "다른 번호를 말할 때는 '선두권에 함께 있다'고만 하고, 뒤따른다거나 추격한다고 말하지 마라." if others else ""))
            else:
                facts["지금 선두권에 있는 번호"] = [f"{n}번({CAP[n]} 모자)" for n in allowed]
                rule = "선두권 번호를 한 개 이상 넣어 지금 상황을 한 문장으로 중계하라. 누가 선두인지, 번호들 사이의 순서는 말하지 말고 '선두권'이라고만 하라."
        else:
            if recall:
                facts["조금 전까지 선두권으로 확인된 번호"] = [f"{n}번" for n in recall]
            rule = ("지금 누가 앞서는지는 확인되지 않았다. 지금 앞선 말을 말하지 말고, 경주 정보와 지난 시간, 구간만으로 한 문장을 만들어라. "
                    + ("확인된 번호는 '조금 전까지'라는 과거형으로만 말할 수 있다." if recall else "번호는 쓰지 마라."))
        user = ("사실: " + json.dumps(facts, ensure_ascii=False) + "\n앞의 두 문장: " + json.dumps(prev_lines[-2:], ensure_ascii=False) + "\n지시: " + rule
                + " 앞 문장과 같은 표현은 피하라. 경마장·경주 번호·거리는 첫 문장이 아니면 되풀이하지 마라. 남은 거리를 말할 때는 '약'을 붙여라.")
        line, source = None, "model"
        for attempt in (0, 1):
            out = ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], LINE, 20261117 + k + 1000 * attempt, 0.5 + 0.2 * attempt, f"{race}:{k}:{attempt}")
            cand = tidy(out["line"]) if out and isinstance(out.get("line"), str) else ""
            if check(cand, mode, allowed, recall, leader, follow_ok):
                line, source = cand, "model" if attempt == 0 else "model_retry"
                break
        if line is None:
            line, source = template(mode, allowed, recall, phase, info, leader, follow_ok), "template"
        prev_lines.append(line)
        if mode == "leader":
            last_leader = {"t": p["t"], "numbers": allowed}
        said = numbers_in(line)
        rows.append({"k": k, "t": p["t"], "elapsed": round(elapsed, 1), "phase": phase, "mode": mode, "identifier": p["statements"], "identifier_leader": p.get("leader"),
                     "identifier_order": sorted(ahead), "allowed": allowed, "leader_given": leader, "follow_allowed": follow_ok, "recall": recall,
                     "line": line, "source": source, "said_numbers": said, "leader_claims": leader_claims(line) if mode == "leader" else [],
                     "says_follow": bool(mode == "leader" and leader is not None and FOLLOW.search(line)), "strip": p.get("strip")})
        print(f"{race} {elapsed:5.1f}s [{mode:9s}] leader {leader} follow {follow_ok} | {line}", flush=True)
    (O / "commentary_v4").mkdir(exist_ok=True)
    (O / "commentary_v4" / f"{race}.json").write_text(json.dumps({"race": race, "info": info, "expected_finish_seconds": round(T, 1), "rows": rows}, ensure_ascii=False, indent=1))
    summary.append((race, len(rows), sum(r["mode"] == "leader" for r in rows), sum(r["leader_given"] is not None for r in rows), sum(r["source"] == "template" for r in rows)))
print("races", len(summary), summary)
try:
    urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/generate", data=json.dumps({"model": MODEL, "keep_alive": 0}).encode(), headers={"Content-Type": "application/json"}), timeout=60).read()
except Exception:
    pass
