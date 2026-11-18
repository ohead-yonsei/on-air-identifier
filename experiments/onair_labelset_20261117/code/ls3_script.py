"""Script demo, step 3: turn the identifier's output into a commentary script that reads like a race call.
usage: ls3_script.py DEMO_DIR [MODEL]      (reads ident.json and racecard.json; writes script.json; standard library only; local Ollama)

The flow under test: (1) no strip, picture only; (2) where the identifier is sure, call the leader and the leading group;
(3) where it is not, talk from confirmed information (race card, clock, distance).
Per time point, in order and without looking ahead, the code decides what may be said ("facts"); a local language model (text only,
it sees no picture; default qwen3.8:27b) words it as a caster would; the code checks the sentence and asks again, or falls back to
a plain pattern, when the check fails.
Facts: the identifier's stated numbers and leader at that moment; whether the others are known to be behind the leader; changes
against the time points before (a new leader, a new number in the leading group); the race card (number -> horse name, distance,
runners); an approximate distance to go, said once per 200 m; at the clock stop the winner call if there is one.
Checks: only allowed numbers and names, every name with its number ("3번 울트라삭스"); a leader only when one is given, and only that
horse; no ranks, margins, rider names; nothing about gaps, duels or a runaway lead (never known); the finish or a win only at the clock stop; "follows / chases" only when the order is known;
distances and times only when given; "우승" only with a winner call; passing words only when the leader changed.
A sentence that passes the rule checks is also read by the same model in a second, separate call that only answers whether it says
more than the facts. The system text carries a small bank of caster phrases with placeholders instead of numbers (a number in an example gets copied)."""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / sys.argv[1]
ENDPOINT, MODEL = "http://127.0.0.1:11435", (sys.argv[2] if len(sys.argv) > 2 else "qwen3.8:27b-q4_K_M")
card = json.loads((D / "racecard.json").read_text())
ident = json.loads((D / "ident.json").read_text())
NAME = {r["number"]: r["name"] for r in card["runners"]}
T = ident["expected_seconds"]
NUM = re.compile(r"(\d+)\s*번")
LEAD = re.compile(r"선두(?:는|에는|로는|에|로)\s*(\d+)\s*번|(\d+)\s*번\s*[가-힣A-Za-z]{0,10}(?:이|가)?[^.!?]{0,14}?(?:선두(?!권)|맨\s*앞|앞장|이끌|이끕|앞서|페이스)")
LEAD_WORD = re.compile(r"선두(?!권)|맨\s*앞|앞장|이끌|이끕|앞서|앞선|앞으로\s*나|치고\s*나|페이스|리드|흐름을")
BANNED = re.compile(r"\d\s*위|마신|코\s*차|목\s*차|기수|조교사|배당|인기|우승\s*후보|직선\s*주로|코너|곡선|포착")
PASS = re.compile(r"역전|추월|제치|넘어서|바뀌|빼앗|내주|뒤집")
CLOSE = re.compile(r"붙어|나란히|주고받|접전|경합|다투|다툼|숨\s*고르|간발|박빙|머리\s*차|바짝|턱밑|코앞|치열|팽팽|독주|단독|지배|압도|여유|바로\s*뒤|맞서|대결|맞대|뿌리치|따돌|벌리|좁히|압박|위협|견제|버티|버텨|앞에\s*두")      # gaps are never known
EARLY_END = re.compile(r"결승선\s*(?:을\s*)?통과|골인|들어옵|들어왔|꺾|이겼|승리|우승|피니시")                                    # only at the clock stop
FOLLOW = re.compile(r"뒤따|추격|쫓|따라붙|뒤를\s*(?:잇|이)|밀고\s*올라|뒤에서|그\s*뒤")                                   # only when the order is known
SYSTEM = """당신은 한국 경마 실황 캐스터다. 주어진 '사실'만으로 지금 장면을 한 문장(길어야 두 문장, 60자 안쪽)으로 중계한다.

지킬 것
- 사실에 없는 것은 말하지 않는다. 말 사이의 거리, 접전 여부, 기수, 순위(몇 위)는 모른다.
- 말은 반드시 '번호+번 이름' 꼴로 부른다. '말할 수 있는 말'에 없는 말은 부르지 않는다.
- 선두는 사실에 '선두'가 있을 때만, 그 말에 대해서만 말한다. 선두가 없으면 '앞서 있다', '이끈다' 같은 말도 쓰지 않고 '선두권'이라고만 한다.
- '뒤따른다', '추격한다'는 사실에 '선두를 뒤따르고 있음'이 있을 때만 쓴다.
- 거리와 시간은 사실에 숫자가 주어졌을 때만 말한다.
- 주로의 위치(코너, 직선주로)는 모른다. 말하지 않는다.
- 결승선 통과, 우승, 누가 누구를 이겼다는 말은 사실에 '결승선을 통과했다'가 있을 때만 한다.
- 구간 이름(초반, 중반, 막판)을 그대로 읽지 않는다. 말투에만 반영한다: 앞쪽은 차분하게, 뒤쪽은 짧고 힘 있게.
- 앞선 문장과 같은 틀을 되풀이하지 않는다.

말투 예시 ([ ] 안은 사실로 바꿔 쓴다. 그대로 베끼지 말고 섞어 쓴다)
- 선두가 그대로일 때: "[선두], 여전히 레이스를 이끕니다." / "[선두]가 선두를 지키며 달려갑니다." / "앞에서는 계속 [선두]."
- 선두 뒤에 다른 말이 있을 때(뒤따르고 있음): "[선두]가 이끌고, 그 뒤를 [말]이 따릅니다." / "선두 [선두], 뒤에는 [말]."
- 선두는 모르고 선두권만 알 때: "선두권에는 [말], [말]이 보입니다." / "[말]과 [말], 선두권에서 달립니다."
- 새 말이 선두권에 보일 때: "[말]도 선두권에 올라옵니다." / "[말]이 선두권에 모습을 드러냅니다."
- 선두가 바뀌었을 때: "선두가 바뀝니다! [새 선두]가 앞으로 나섭니다." / "[새 선두], [전 선두]를 넘어 선두로!"
- 거리가 주어졌을 때: "결승선까지 [거리]." / "이제 [거리] 남았습니다."
- 선두권을 확인하지 못했을 때: "결승선까지 [거리], 승부는 아직 알 수 없습니다." / "[경주] 달려가고 있습니다."
- 결승선, 우승마를 불렀을 때: "결승선 통과! [우승마], 가장 먼저 들어옵니다!" / "[우승마] 우승!"
- 결승선, 우승마를 못 불렀을 때: "결승선을 통과합니다! 우승마는 공식 결과를 기다려 보겠습니다."

JSON으로만 답한다."""
SCHEMA = {"type": "object", "properties": {"line": {"type": "string"}}, "required": ["line"], "additionalProperties": False}


def who(n):
    return f"{n}번 {NAME.get(n, '')}".strip()


def ask(user, seed, temperature):
    body = json.dumps({"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "format": SCHEMA, "stream": False,
                       "think": False, "keep_alive": "10m", "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192, "num_predict": 200, "top_p": 0.9}}).encode()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/chat", data=body, headers={"Content-Type": "application/json"}), timeout=600) as r:
            line = json.loads(json.loads(r.read().decode())["message"]["content"]).get("line", "").strip()
    except Exception as e:      # noqa: BLE001
        line = ""
    return re.sub(r"(\d)\s+(번|초|m\b|미터)", r"\1\2", line), round(time.perf_counter() - t0, 2)


def josa(word, a, b):
    """이/가, 을/를 by the last syllable."""
    c = ord(word[-1]) - 0xAC00
    return a if 0 <= c < 11172 and c % 28 else b


JUDGE_SYSTEM = ("당신은 경마 중계 문장의 사실 검수자다. '사실'과 '문장'을 받는다. 문장이 사실에 없는 내용을 하나라도 말하면 extra를 true로 한다. "
                "특히 다음을 본다: 사실과 반대인 앞뒤 순서, 말 사이의 거리나 압박·경합, 사실에 없는 말·선두·거리·시간, 결승이나 승패. "
                "말투, 감탄, '달린다'·'질주한다'·'이어간다' 같은 꾸밈은 내용이 아니므로 괜찮다. JSON으로만 답한다.")
JUDGE_SCHEMA = {"type": "object", "properties": {"extra": {"type": "boolean"}, "what": {"type": "string"}}, "required": ["extra", "what"], "additionalProperties": False}


def judge(facts, line, seed):
    """Second pass by the same local model: does the sentence say anything the facts do not?"""
    body = json.dumps({"model": MODEL, "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": "사실\n- " + "\n- ".join(facts) + "\n\n문장\n" + line}],
                       "format": JUDGE_SCHEMA, "stream": False, "think": False, "keep_alive": "10m", "options": {"temperature": 0, "seed": seed, "num_ctx": 8192, "num_predict": 120}}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/api/chat", data=body, headers={"Content-Type": "application/json"}), timeout=600) as r:
            out = json.loads(json.loads(r.read().decode())["message"]["content"])
        return (out.get("what") or "says more than the facts") if out.get("extra") else None
    except Exception:      # noqa: BLE001
        return None


def check(line, f):
    if not line or len(line) > 100 or BANNED.search(line):
        return "empty, too long or a banned word"
    if line in f.get("previous", []):
        return "the same sentence as just before"
    nums = {int(n) for n in NUM.findall(line)}
    if not nums <= set(f["allowed"]):
        return "a number that is not allowed"
    for n, name in NAME.items():
        c = line.count(name)
        if c and (n not in f["allowed"] or line.count(f"{n}번 {name}") != c):
            return "a name without its number, or a horse that is not allowed"
    if f["mode"] == "ranking" and not nums:
        return "no number in a ranking sentence"
    if not f.get("finish"):
        claimed = {int(g) for m in LEAD.finditer(line) for g in m.groups() if g}
        if f["leader"] is None:
            if LEAD_WORD.search(line):
                return "a leader word without a given leader"
        elif not claimed <= {f["leader"]} | (set(f["change"] or []) if PASS.search(line) else set()):
            return "leader claimed for another horse"
        if FOLLOW.search(line) and not f.get("follow_ok"):
            return "following words without a known order"
    if f.get("finish"):
        if "우승" in line.replace("우승마는", "") and not f.get("winner"):
            return "winner without a winner call"
    elif EARLY_END.search(line):
        return "the finish or a win before the clock stopped"
    if PASS.search(line) and not f.get("change"):
        return "passing words without a change of leader"
    if CLOSE.search(line):
        return "a word about gaps or a duel"
    if re.search(r"\d+\s*초", line) or re.search(r"[초중]반|막판", line) or (re.search(r"결승\s*직전|결승선\s*직전|결승점", line) and f["phase"] not in ("결승선 직전", "결승")):
        return "a time or a phase name read out"
    said_m = {int(x) for x in re.findall(r"(\d+)\s*(?:m\b|미터|M\b)", line)}
    if said_m - ({f["mark"]} if f.get("say_distance") else set()) - ({card["distance"]} if f.get("early") else set()):
        return "a distance that was not given"
    return None


def template(f):
    if f.get("finish"):
        return f"결승선 통과! {who(f['winner'])}, 가장 먼저 들어옵니다!" if f.get("winner") else "결승선을 통과합니다! 우승마는 공식 결과를 기다려 보겠습니다."
    dist = f"결승선까지 약 {f['mark']}m. " if f.get("say_distance") else ""
    if f["leader"] is not None:
        return dist + f"선두는 {who(f['leader'])}" + (", 그 뒤를 " + ", ".join(who(n) for n in f["group"]) + josa(who(f["group"][-1]), "이", "가") + " 따릅니다." if f["group"] and f["follow_ok"] else
                                                    (", 선두권에 " + ", ".join(who(n) for n in f["group"]) + "도 있습니다." if f["group"] else "입니다."))
    if f["group"]:
        return dist + "선두권에 " + ", ".join(who(n) for n in f["group"]) + josa(who(f["group"][-1]), "이", "가") + " 보입니다."
    return f"{card['place']} {card['race']}경주 {card['distance']}m, " + (f"결승선까지 약 {f['mark']}m 남았습니다." if f.get("say_distance") else "달려가고 있습니다.")


rows, prev_lines, last_leader, recent, last_mark = [], [], None, [], None
for k, p in enumerate(ident["points"]):
    e = p["elapsed"]
    to_go = max(0, card["distance"] * (1 - e / T))
    phase = "출발 직후" if e < 12 else "초반" if e < 0.3 * T else "중반" if e < T - 36 else "막판" if e < T - 12 else "결승선 직전"
    said, lead = list(p["statements"]), p["leader"]
    finish, early = p.get("at_clock_stop", False), p.get("early", False)
    f = {"elapsed": e, "phase": "결승" if finish else phase, "leader": lead, "group": [n for n in said if n != lead], "finish": finish, "early": early,
         "winner": None, "change": None, "new": [], "follow_ok": False}
    mark = int(-(-to_go // 200) * 200)                 # distance: said once per 200 m
    f["mark"] = mark
    if finish:
        f["winner"] = ident["winner_call"]["number"]
        f.update(leader=None, group=[], allowed=[f["winner"]] if f["winner"] else [], mode="finish", say_distance=False)
    else:
        if lead is not None and last_leader is not None and lead != last_leader["number"] and p["t"] - last_leader["t"] <= 9:
            f["change"] = [last_leader["number"], lead]
        f["new"] = [n for n in said if n != lead and all(n not in s for s in recent[-2:])] if len(recent) >= 2 and said else []
        ahead = {(o["ahead"], o["behind"]) for o in p.get("order", [])}
        f["follow_ok"] = bool(lead is not None and f["group"] and all((lead, n) in ahead for n in f["group"]))
        f["mode"] = "ranking" if (said or lead is not None) else "confirmed"
        f["allowed"] = sorted(set(said) | ({lead} if lead is not None else set()) | set(f["change"] or []))
        f["say_distance"] = bool(e >= 9 and 200 <= mark <= card["distance"] - 200 and (mark != last_mark or f["mode"] == "confirmed"))
        if f["say_distance"]:
            last_mark = mark
    facts = [f"경주: {card['place']} {card['race']}경주 {card['distance']}m" + (f", {len(card['runners'])}두 출전" if early else ""), f"말투: {f['phase']}"]
    if early:
        facts.append("방금 출발했다. 어느 말이 앞서는지는 아직 확인하지 않았다." if e <= 3 else "출발 직후 자리를 잡는 구간이다. 어느 말이 앞서는지는 아직 확인하지 않았다.")
    elif finish:
        facts.append(f"결승선을 통과했다. 화면으로 판단한 우승마: {who(f['winner'])}" if f["winner"] else "결승선을 통과했다. 우승마는 화면으로 확정하지 못했다.")
    else:
        if f["say_distance"]:
            facts.append(f"거리: 결승선까지 약 {mark}m")
        facts.append(f"선두: {who(lead)}" if lead is not None else "선두: 확인되지 않음")
        if f["group"]:
            facts.append("선두권의 다른 말: " + ", ".join(who(n) for n in f["group"]) + (" (선두를 뒤따르고 있음)" if f["follow_ok"] else " (앞뒤 순서는 모름)"))
        if f["change"]:
            facts.append(f"변화: 선두가 {who(f['change'][0])}에서 {who(f['change'][1])}(으)로 바뀌었다")
        if f["new"]:
            facts.append("변화: " + ", ".join(who(n) for n in f["new"]) + "가 선두권에 새로 보인다")
        if f["mode"] == "confirmed":
            facts.append("지금은 선두권을 확인하지 못했다. 어떤 말도 부르지 않는다.")
    facts.append("말할 수 있는 말: " + (", ".join(who(n) for n in f["allowed"]) if f["allowed"] else "없음"))
    user = "사실\n- " + "\n- ".join(facts) + ("\n\n앞선 중계 (이어지는 문장이다. 같은 틀을 되풀이하지 말 것)\n- " + "\n- ".join(prev_lines[-3:]) if prev_lines else "") + "\n\n지금 장면의 중계를 써라."
    f["previous"] = prev_lines[-3:]
    line, source, tries = "", "template", []
    for attempt in range(4):
        cand, sec = ask(user, 20261118 + 7 * k + attempt, 0.6 + 0.1 * attempt)
        why = check(cand, f)
        if why is None:
            extra = judge([x for x in facts if not x.startswith("말투")], cand, 20261118 + k)
            why = None if extra is None else "the checking model: " + extra
        tries.append({"line": cand, "rejected": why, "seconds": sec})
        if why is None:
            line, source = cand, "model" if attempt == 0 else "model, asked again"
            break
    if not line:
        line = template(f)
    rows.append({"k": k, "t": p["t"], "elapsed": e, "frame": p["frame"], "mode": f["mode"], "phase": f["phase"], "facts": facts,
                 "given": {x: f.get(x) for x in ("leader", "group", "change", "new", "winner", "allowed", "follow_ok", "say_distance", "mark")},
                 "line": line, "source": source, "tries": tries, "identifier": {x: p[x] for x in ("statements", "leader", "leader_source", "leader_model", "order")}})
    prev_lines.append(line)
    recent.append(said)
    if lead is not None:
        last_leader = {"number": lead, "t": p["t"]}
    print(f"{int(e):3d}s [{f['mode']}/{source}/{tries[-1]['seconds']}s] {line}" + ("   <- rejected first: " + "; ".join(f"{x['line']} ({x['rejected']})" for x in tries if x["rejected"]) if len(tries) > 1 else ""), flush=True)
sec = [x["seconds"] for r in rows for x in r["tries"]]
(D / "script.json").write_text(json.dumps({"racecard": card, "model": MODEL, "system": SYSTEM, "rows": rows, "made": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                          "model_calls": len(sec), "seconds_per_call_median": sorted(sec)[len(sec) // 2]}, ensure_ascii=False, indent=1))
print("sources:", {s: sum(r["source"] == s for r in rows) for s in ("model", "model, asked again", "template")}, "| median seconds per call", sorted(sec)[len(sec) // 2])
