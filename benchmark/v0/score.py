"""ON-AIR draft benchmark v0: scorer.
usage: python3 benchmark/v0/score.py PREDICTIONS.jsonl [--split test|dev|all] [--track strip|official|all] [--part PART] [--out SCORE.json]
       (standard library only)

Two tracks (the official track was added 2026-11-19):
  strip      items.jsonl, the moments the broadcaster's running-order strip is on screen; the answer is the strip (default)
  official   items_official.jsonl, the moments the strip does not cover (the first seconds, the last half minute, the finish itself);
             the answer is the order worked out from the official passing times. --part start|middle|finish|finish_itself picks a part.
             Next to every strict count there is a lenient one (ending _lenient): a number also counts as right when the horse is within
             3 m of the third, a leader when within 3 m of the first, because the official times are given to 0.1 s (about 1.7 m).
             On the strip track the lenient counts equal the strict ones.

PREDICTIONS.jsonl: one line per item, {"id": ..., "text": "한국어 중계 한 문장"}.
Horses are referred to by number, written like "4번". Optional fields override what is read from the text:
"numbers": [..] (the numbers the sentence puts in the leading group), "leader": n or null (the number it calls the leader).
English sentences (added 2026-11-19): a line with "lang": "en" is read with the English rules; horses are written "No. 4".
Several horses said to lead together ("No. 5 and No. 6 lead") are all read as leader claims.
Items without a line count as "said nothing".

Headline pair (added 2026-11-18; a system that only ever says safe things scores 0 on the first, a system that guesses scores high on the second):
  leader right      items whose sentence names a leader, and every leader it names is the strip's first / all items   (higher is better)
  leader wrong      items whose sentence names a leader that is not the strip's first / all items                     (lower is better)
  The rest of the items carry no leader claim.
Also added: right and clean = items with at least one right number and no wrong number / all items (higher is better).

What else is counted (the answer is the running-order strip at that moment):
  wrong number      a stated number that is not among the strip's first three
  hallucination     share of items whose sentence contains at least one wrong number             (lower is better)
  A (precision)     stated numbers that are among the first three / all stated numbers           (higher is better)
  B (coverage)      items with at least one right number / all items                             (higher is better; says how much was told)
  leader            leader claims that name the strip's first / all leader claims, and how many items carry a leader claim
  not in the race   stated numbers that do not run in this race at all
  fourth or fifth   wrong numbers that are fourth or fifth on the strip (near misses)
Intervals: bootstrap over races, 2000 resamples, seed 20261118."""
import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUMBER = re.compile(r"(\d+)\s*번")
LEADER = re.compile(r"선두는\s*(?:[가-힣 ]{0,12}모자\s*)?(\d+)\s*번|선두(?:에|로)\s*(?:[가-힣 ]{0,12}모자\s*)?(\d+)\s*번|(\d+)\s*번이\s*선두(?!권)|(\d+)\s*번이\s*(?:맨\s*)?앞(?:장|서)|(\d+)\s*번이\s*(?:경주를\s*)?이끌")


_NN = r"(?<!race )(?:\bNo\.?|\bnumber|#)\s*\d+"          # one horse, English ("Race No. 3" is the race, not a horse)
_LIST = rf"(?:{_NN}\s*(?:,\s*(?:and\s+)?|\s+and\s+))*{_NN}"          # "No. 3", "No. 3 and No. 5", "No. 3, No. 5 and No. 7"
_GROUP = r"(?:group|pack|bunch|trio|pair|runners|horses)"
NUMBER_EN = re.compile(r"(?<!race )(?:\bNo\.?|\bnumber|#)\s*(\d+)", re.I)
NOS_EN = re.compile(r"\bNos\.?\s*(\d+(?:\s*,\s*\d+)*(?:,?\s*and\s*\d+)?)", re.I)          # "Nos. 3 and 6" -> "No. 3 and No. 6"
LEADER_EN = re.compile(
    rf"(?P<a>{_LIST}),?\s+(?:(?:is|are|has|have|who|still|now|just|both|all|also|too|together|alongside|level|remains?|stays?|continues?|currently|out|up|right|there|clearly|firmly|narrowly|comfortably|to)\s+){{0,3}}"
    rf"(?:leads?\b(?!\s+{_GROUP})|leading(?!\s+{_GROUP})|in\s+front(?!\s+(?:of|{_GROUP}))|in\s+the\s+lead(?!ing|\s+{_GROUP})|at\s+the\s+front(?!\s+of\s+the\s+leading)|"
    r"at\s+the\s+head|on\s+the\s+lead|sets?\s+the\s+pace|setting\s+the\s+pace|shows?\s+the\s+way|showing\s+the\s+way|heads?\s+the\s+(?:field|pack|leading\s+group|front\s+group)|heading\s+the\s+field|"
    r"(?:takes?|taken|took|holds?|has|have|keeps?|grabs?|taking|holding|keeping)\s+(?:over\s+|up\s+)?the\s+lead|hits?\s+the\s+front|the\s+leaders?\b|the\s+front[- ]runners?)"
    rf"|(?:leaders?(?:\s+is|\s+are|,|:)?|led\s+by|the\s+lead\s+(?:is\s+)?(?:with|held\s+by|goes\s+to|belongs\s+to)|(?:in\s+front|the\s+lead|the\s+way)\s+(?:is|are|it'?s)|front[- ]runners?(?:\s+is|\s+are|,)?|"
    rf"leading\s+(?:is|are|it'?s))\s+(?P<b>{_LIST})",
    re.I)


# ---- rules v1 for Korean leader claims (2026-11-22, score1/DESIGN_KO.md). The default stays v0: set RULES = "v1" or give --rules v1 ----
RULES = "v0"
_SEG = re.compile(r"[.!?,]+")
_NOT_NAME = r"(?!선두|앞|맨|가장|우승|현재|지금|계속|여전히|다시|단독|크게|근소|살짝|바로|역시|먼저|안쪽|바깥|뒤|서울|출발|경주)"
_NAME = rf"(?:\s+{_NOT_NAME}[가-힣]{{2,10}}?)?"
_SUBJ = re.compile(rf"(\d+)\s*번{_NAME}\s*(?:이|가|은|는)(?:(?![가-힣])|(?=선두|앞|맨|가장))")
_ALSO = re.compile(rf"(\d+)\s*번{_NAME}\s*도(?![가-힣])")
_OBJ_AFTER = re.compile(r"\s*(?:을|를|보다|에게|의|도|과|와|하고|랑|까지|마저)(?![가-힣])")
_JOINED_BEFORE = re.compile(rf"\d+\s*번{_NAME}\s*(?:과|와|하고|이랑|랑|및|그리고)\s*$")
_ONLY_NUMBER = re.compile(r"\s*(\d+)\s*번(?:\s+[가-힣]{2,10})?\s*")
_LEAD = re.compile(r"선두(?!권|\s*(?:그룹|무리|집단))|앞서(?!\s*(?:가는|가던|달리는|있는|나가는)\s*(?:\d|말|무리|선두))|앞선\s*(?:가운데|채|상황|상태)|앞섭|앞장|앞을\s*달리|맨\s*앞|이끌|이끕|리드|가장\s*먼저|(?<!\d)1\s*위|(?<!\d)1\s*착|우승")
_CHASE = re.compile(r"추격|쫓|따라가|따라붙|따라잡|따르|뒤를|뒤에서|뒤로|바로\s*뒤|노리|노려|위협|압박|넘보|도전|격차|좁히|줄이|다툼|다투|다툽|경쟁|경합|접전|싸움|맞붙|내주|내줬|빼앗|밀려|놓치|놓쳤")
_TOGETHER = re.compile(r"나란히|함께|같이|동시에")
_MODIFIES_NEXT = re.compile(r"\s*(?:의|인|에\s*선|에\s*있는|를\s*달리는|를\s*지키는|를\s*이끄는)?\s*")
_NUM_THEN_LEAD = re.compile(r"(\d+)\s*번\s*선두(?!권|\s*(?:그룹|무리|집단))")


_NOUN_LEAD = re.compile(r"선두|맨\s*앞|1\s*위|1\s*착")
_NEXT_VERB = re.compile(r"(?:고|며|면서|는데|지만)(?=\s|$)")


def _clean(before, after, word="선두"):
    """No word for chasing, sharing or losing the lead before the leading word; and, when the leading word is a noun ("선두를 추격",
    "선두에서 나란히", "선두 다툼"), none in the rest of its own verb phrase either."""
    if _CHASE.search(before) or _TOGETHER.search(before):
        return False
    if not _NOUN_LEAD.fullmatch(word):
        return True
    cut = _NEXT_VERB.search(after)
    own = after[:cut.start()] if cut else after
    return not (_CHASE.search(own) or _TOGETHER.search(own))


def leaders_ko_v1(text):
    """Numbers a Korean sentence names as the one horse in front, read piece by piece (pieces end at punctuation).
    A number counts when it is the subject of a word for leading ("3번이 크게 앞서고"), stands right before "선두" ("3번 선두"), or is
    the only number of a piece that says who leads ("선두는 3번"); not when the piece speaks of chasing, of two horses together, or of
    the leading group."""
    out, carry, prev_only = set(), None, False          # carry: the number of a piece that is only "N번 (name)"
    for seg in _SEG.split(text):
        if not seg or not seg.strip():
            continue
        only = _ONLY_NUMBER.fullmatch(seg)
        if only:
            carry, prev_only = (None if prev_only else int(only.group(1))), True          # "4번, 3번, ..." is a list: no single subject
            continue
        nums = [(m.start(), m.end()) for m in NUMBER.finditer(seg)]
        subjects = list(_SUBJ.finditer(seg))
        stops = sorted([m.start() for m in subjects] + [m.start() for m in _ALSO.finditer(seg)])
        for sm in subjects:
            if _JOINED_BEFORE.search(seg[:sm.start()]) or (prev_only and not seg[:sm.start()].strip()):
                continue                                                                  # "3번과 4번이", "3번, 4번이": not one horse
            end = next((p for p in stops if p > sm.start()), len(seg))
            window = seg[sm.end():end]
            for lm in _LEAD.finditer(window):
                before, after = window[:lm.start()], window[lm.end():]
                nxt = NUMBER.search(after)
                cut = after[:nxt.start()] if nxt else after
                if nxt and _MODIFIES_NEXT.fullmatch(cut):
                    continue                                                              # "선두 3번을 ...": the leading word belongs to the next number
                if _clean(before, cut, lm.group(0)):
                    out.add(int(sm.group(1)))
                break
        for m in _NUM_THEN_LEAD.finditer(seg):                                            # "3번 선두", "9번선두"
            after = seg[m.end():]
            nxt = NUMBER.search(after)
            if not _JOINED_BEFORE.search(seg[:m.start()]) and _clean("", after[:nxt.start()] if nxt else after):
                out.add(int(m.group(1)))
        lead = _LEAD.search(seg)
        if lead and not subjects and not (_CHASE.search(seg) or _TOGETHER.search(seg)):
            if not nums and carry is not None:
                out.add(carry)                                                            # "2번 용비태양, 선두를 지킵니다"
            elif len(nums) == 1 and not _OBJ_AFTER.match(seg[nums[0][1]:]) and not _NUM_THEN_LEAD.search(seg):
                out.add(int(NUMBER.search(seg).group(1)))                                 # "선두는 3번", "우승은 8번"
        carry, prev_only = None, False
    return sorted(out)


def claims(pred, rules=None):
    text = (pred or {}).get("text") or ""
    if (pred or {}).get("lang") == "en":
        text = NOS_EN.sub(lambda m: " and ".join("No. " + n for n in re.findall(r"\d+", m.group(1))), text)
        found = sorted({int(n) for n in NUMBER_EN.findall(text)})
        leaders = sorted({int(n) for m in LEADER_EN.finditer(text) for n in NUMBER_EN.findall(m.group("a") or m.group("b"))})
    else:
        found = sorted({int(n) for n in NUMBER.findall(text)})
        leaders = leaders_ko_v1(text) if (rules or RULES) == "v1" else sorted({int(g) for m in LEADER.finditer(text) for g in m.groups() if g})
    nums = pred["numbers"] if pred and "numbers" in pred else found
    if pred and "leader" in pred:
        lead = [pred["leader"]] if pred["leader"] is not None else []
    else:
        lead = leaders
    return sorted(set(nums) | set(lead)), lead


def answer_key(track):
    """[(item, answer)], answer = {"top3", "leader", "second", "near", "top3_lenient", "leader_lenient", "part", "to_finish"}."""
    read = lambda name: [json.loads(l) for l in (HERE / name).read_text().splitlines()]
    out = []
    if track in ("strip", "all"):
        ans = {a["id"]: a["strip"] for a in read("answers.jsonl")}
        for it in read("items.jsonl"):
            s = ans[it["id"]]
            out.append((it, {"top3": s[:3], "leader": s[0], "second": s[1], "near": s[3:], "top3_lenient": s[:3], "leader_lenient": s[:1], "part": "strip", "to_finish": None}))
    if track in ("official", "all"):
        ans = {a["id"]: a for a in read("answers_official.jsonl")}
        for it in read("items_official.jsonl"):
            a = ans[it["id"]]
            out.append((it, {"top3": a["top3"], "leader": a["leader"], "second": a["order"][1], "near": a["order"][3:5], "top3_lenient": a["top3_lenient"],
                             "leader_lenient": a["leader_lenient"], "part": a["part"], "to_finish": a["to_finish"]}))
    return out


def rows(pred_path, split, track="strip", part=None):
    races = json.loads((HERE / "races.json").read_text())["races"]
    pred = {p["id"]: p for p in map(json.loads, Path(pred_path).read_text().splitlines()) if p}
    out = []
    for it, ans in answer_key(track):
        if split != "all" and it["split"] != split:
            continue
        if part and not (ans["to_finish"] == 0 if part == "finish_itself" else ans["part"] == part):
            continue
        runners = set(races[it["race_key"]]["runners"])
        nums, lead = claims(pred.get(it["id"]))
        out.append({"id": it["id"], "race": it["race_key"], "said": len(nums), "right": sum(n in ans["top3"] for n in nums),
                    "near": sum(n in ans["near"] for n in nums), "absent": sum(n not in runners for n in nums) if runners else 0,
                    "lead": len(lead), "lead_right": sum(n == ans["leader"] for n in lead), "lead_second": sum(n == ans["second"] for n in lead),
                    "right_lenient": sum(n in ans["top3_lenient"] for n in nums), "lead_right_lenient": sum(n in ans["leader_lenient"] for n in lead),
                    "part": ans["part"], "to_finish": ans["to_finish"], "answered": it["id"] in pred})
    return out


def summarise(rs):
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    said, n = sum(r["said"] for r in rs), len(rs)
    s = {"items": n, "items_with_a_number": sum(r["said"] > 0 for r in rs), "numbers": said, "numbers_right": sum(r["right"] for r in rs),
         "numbers_fourth_or_fifth": sum(r["near"] for r in rs), "numbers_not_in_race": sum(r["absent"] for r in rs),
         "items_with_wrong_number": sum(r["said"] > r["right"] for r in rs), "items_with_right_number": sum(r["right"] > 0 for r in rs),
         "leader_claims": sum(r["lead"] for r in rs), "leader_right": sum(r["lead_right"] for r in rs), "leader_second": sum(r["lead_second"] for r in rs),
         "items_with_leader_claim": sum(r["lead"] > 0 for r in rs),
         "items_leader_right": sum(r["lead"] > 0 and r["lead_right"] == r["lead"] for r in rs),
         "items_leader_wrong": sum(r["lead"] > r["lead_right"] for r in rs),
         "items_right_and_clean": sum(r["right"] > 0 and r["said"] == r["right"] for r in rs)}
    s.update(hallucination=pct(s["items_with_wrong_number"], n), A=pct(s["numbers_right"], said), B=pct(s["items_with_right_number"], n),
             leader=pct(s["leader_right"], s["leader_claims"]), leader_coverage=pct(s["items_with_leader_claim"], n),
             leader_right_share=pct(s["items_leader_right"], n), leader_wrong_share=pct(s["items_leader_wrong"], n),
             right_and_clean=pct(s["items_right_and_clean"], n))
    s.update(numbers_right_lenient=sum(r["right_lenient"] for r in rs), items_with_wrong_number_lenient=sum(r["said"] > r["right_lenient"] for r in rs),
             items_leader_right_lenient=sum(r["lead"] > 0 and r["lead_right_lenient"] == r["lead"] for r in rs),
             items_leader_wrong_lenient=sum(r["lead"] > r["lead_right_lenient"] for r in rs))
    s.update(A_lenient=pct(s["numbers_right_lenient"], said), hallucination_lenient=pct(s["items_with_wrong_number_lenient"], n),
             leader_right_share_lenient=pct(s["items_leader_right_lenient"], n), leader_wrong_share_lenient=pct(s["items_leader_wrong_lenient"], n))
    return s


KEYS = ("leader_right_share", "leader_wrong_share", "right_and_clean", "hallucination", "A", "B", "leader")


def interval(rs, key, n=2000, seed=20261118):
    races = sorted({r["race"] for r in rs})
    by = {k: [r for r in rs if r["race"] == k] for k in races}
    rng, vals = random.Random(seed), []
    for _ in range(n):
        v = summarise([x for k in rng.choices(races, k=len(races)) for x in by[k]])[key]
        if v is not None:
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--split", default="test", choices=("test", "dev", "all"))
    ap.add_argument("--track", default="strip", choices=("strip", "official", "all"))
    ap.add_argument("--part", choices=("strip", "start", "middle", "finish", "finish_itself"))
    ap.add_argument("--out")
    ap.add_argument("--rules", default="v0", choices=("v0", "v1"), help="how Korean leader claims are read: v0 = the published tables; v1 = wider (score1/RESULT_KO.md)")
    ap.add_argument("--only-answered", action="store_true", help="score only the items that have a line (for a run that is not finished)")
    a = ap.parse_args()
    globals()["RULES"] = a.rules
    rs = rows(a.predictions, a.split, a.track, a.part)
    if a.only_answered:
        rs = [r for r in rs if r["answered"]]
    s = summarise(rs)
    s["interval"] = {k: interval(rs, k) for k in KEYS}
    s.update(split=a.split, track=a.track, part=a.part, races=len({r["race"] for r in rs}), predictions=a.predictions, items_without_a_line=sum(not r["answered"] for r in rs))
    print(f"{a.predictions}  [{a.split}, {a.track} track{', ' + a.part if a.part else ''}: {s['races']} races, {s['items']} items, {s['items_without_a_line']} without a line]")
    print(f"  LEADER RIGHT  items naming the strip's first  {s['leader_right_share']}%  ({s['items_leader_right']}/{s['items']})  {s['interval']['leader_right_share']}")
    print(f"  LEADER WRONG  items naming another leader   {s['leader_wrong_share']}%  ({s['items_leader_wrong']}/{s['items']})  {s['interval']['leader_wrong_share']}")
    print(f"  right and clean (a right number, no wrong)  {s['right_and_clean']}%  ({s['items_right_and_clean']}/{s['items']})  {s['interval']['right_and_clean']}")
    print(f"  hallucination (items with a wrong number)  {s['hallucination']}%  ({s['items_with_wrong_number']}/{s['items']})  {s['interval']['hallucination']}")
    print(f"  A  right numbers / stated numbers          {s['A']}%  ({s['numbers_right']}/{s['numbers']})  {s['interval']['A']}")
    print(f"  B  items with a right number               {s['B']}%  ({s['items_with_right_number']}/{s['items']})  {s['interval']['B']}")
    print(f"  leader claims right                        {s['leader']}%  ({s['leader_right']}/{s['leader_claims']})  {s['interval']['leader']}   in {s['leader_coverage']}% of items; second {s['leader_second']}")
    print(f"  wrong numbers: fourth or fifth {s['numbers_fourth_or_fifth']}, not in the race {s['numbers_not_in_race']}, other {s['numbers'] - s['numbers_right'] - s['numbers_fourth_or_fifth'] - s['numbers_not_in_race']}")
    if a.track != "strip":
        print(f"  lenient (within 3 m): leader right {s['leader_right_share_lenient']}%  leader wrong {s['leader_wrong_share_lenient']}%  hallucination {s['hallucination_lenient']}%  A {s['A_lenient']}%")
    if a.out:
        Path(a.out).write_text(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
