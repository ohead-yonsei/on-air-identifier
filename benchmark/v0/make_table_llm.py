"""ON-AIR draft benchmark v0: tables for the baselines whose sentence is written by a language model, Korean and English.
usage: python3 benchmark/v0/make_table_llm.py [--split test] [--prefix ours_v11]        (standard library only)

Rows per language: the fixed sentence pattern, the model's first attempt with no check ("raw"), the checked sentence ("guarded").
Files (baselines/): PREFIX_template_LANG[_official].jsonl, PREFIX_llm_LANG[_official].jsonl and ..._raw.jsonl; for Korean the pattern row
is the existing ours_identifier_v11[_official].jsonl. Scored on the strip track, the official track and both together.
Also: where the checked sentence came from (model, second attempt, pattern) and how the first attempt failed the check.
Writes results_llm.json and fills README_KO.md between the results-llm marks."""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--prefix", default="ours_v11")
ap.add_argument("--no-readme", action="store_true")
a = ap.parse_args()
B = HERE / "baselines"
ROWS = [("ko", "template", "ours_identifier_v11", "한국어, 정해진 문장 틀"),
        ("ko", "raw", f"{a.prefix}_llm_ko", "한국어, 모델이 쓴 문장 (검사 없음)"),
        ("ko", "guarded", f"{a.prefix}_llm_ko", "**한국어, 모델이 쓴 문장 + 검사**"),
        ("en", "template", f"{a.prefix}_template_en", "영어, 정해진 문장 틀"),
        ("en", "raw", f"{a.prefix}_llm_en", "영어, 모델이 쓴 문장 (검사 없음)"),
        ("en", "guarded", f"{a.prefix}_llm_en", "**영어, 모델이 쓴 문장 + 검사**")]
pc = lambda v: "—" if v is None else f"{v:.1f}%"
KO = {"leader not said": "준 선두를 채점 규칙이 읽지 못함", "other figure": "읽을 수 없는 숫자 표기", "other script": "다른 문자가 섞임", "missing number": "번호가 빠짐",
      "extra number": "주지 않은 번호", "wrong distance": "거리가 틀림", "wrong time": "시간이 틀림", "wrong race number": "경주 번호가 틀림", "number in words": "숫자를 글자로 씀",
      "leader said, none given": "주지 않은 선두를 말함", "other leader": "다른 말도 선두라고 함", "word": "주지 않은 주장의 낱말", "empty": "빈 문장", "too long": "너무 김",
      "speaks of the lead, nothing given": "아무것도 주지 않았는데 선두 이야기"}


def files(name, kind):
    suffix = "_raw" if kind == "raw" else ""
    return B / f"{name}{suffix}.jsonl", B / f"{name}_official{suffix}.jsonl"


def measure(path, track):
    rs = [r for r in score.rows(path, a.split, track) if r["answered"]]
    s = score.summarise(rs)
    s["interval"] = {k: score.interval(rs, k) for k in ("leader_right_share", "leader_wrong_share", "hallucination")}
    return s


res, t1, t2 = {}, ["| 문장 | 문항 | 선두를 맞게 ↑ | 선두를 틀리게 ↓ | 틀린 문장 비율 ↓ | A ↑ | B ↑ | 띠 문항만<br>맞게 / 틀리게 | 띠 없는 문항만<br>맞게 / 틀리게 (넉넉한 정답) |", "|---|---|---|---|---|---|---|---|---|"], []
for lang, kind, name, label in ROWS:
    f_strip, f_off = files(name, kind)
    if not (f_strip.exists() and f_off.exists()):
        continue
    both = B / f".both_{name}_{kind}.jsonl"
    both.write_text(f_strip.read_text() + f_off.read_text())
    s, ss, so = measure(both, "all"), measure(f_strip, "strip"), measure(f_off, "official")
    both.unlink()
    res[f"{lang} {kind}"] = {"both": s, "strip": ss, "official": so}
    iv = lambda k: f" ({s['interval'][k][0]:.0f}~{s['interval'][k][1]:.0f})"
    t1.append(f"| {label} | {s['items']} | {pc(s['leader_right_share'])}{iv('leader_right_share')} | {pc(s['leader_wrong_share'])}{iv('leader_wrong_share')} | {pc(s['hallucination'])}{iv('hallucination')} "
              f"| {pc(s['A'])} | {pc(s['B'])} | {pc(ss['leader_right_share'])} / {pc(ss['leader_wrong_share'])} | {pc(so['leader_right_share'])} / {pc(so['leader_wrong_share'])} ({pc(so['leader_wrong_share_lenient'])}) |")
t2 = ["| 언어 | 모델에 맡긴 문항 | 첫 문장이 검사 통과 | 두 번째 문장이 통과 | 문장 틀로 대신함 | 첫 문장이 걸린 까닭 (많은 순) |", "|---|---|---|---|---|---|"]
for lang in ("ko", "en"):
    f_strip, f_off = files(f"{a.prefix}_llm_{lang}", "guarded")
    if not (f_strip.exists() and f_off.exists()):
        continue
    ids = {it["id"] for it, _ in score.answer_key("all") if a.split in ("all", it["split"])}
    every = [r for f in (f_strip, f_off) for r in map(json.loads, f.read_text().splitlines()) if r["id"] in ids]
    rows = [r for r in every if r["source"] != "pattern, nothing given"]          # the writer is not asked when the identifier gives nothing
    src = collections.Counter(r["source"] for r in rows)
    why = collections.Counter(KO.get(re.sub(r"^word: .*", "word", r["raw_fails"]), r["raw_fails"]) for r in rows if r.get("raw_fails"))
    words = collections.Counter(r["raw_fails"][6:] for r in rows if (r.get("raw_fails") or "").startswith("word: "))
    n = len(rows)
    res[f"{lang} sources"] = {"items": len(every), "items_given_to_the_writer": n, "source": dict(src), "raw_fails": dict(why), "words": dict(words.most_common(12))}
    t2.append(f"| {'한국어' if lang == 'ko' else '영어'} | {n} | {pc(100 * src['model'] / n)} | {pc(100 * src['model_retry'] / n)} | {pc(100 * src['template'] / n)} | " + ", ".join(f"{k} {v}" for k, v in why.most_common(6)) + " |")
table = ("**문장을 누가 쓰느냐에 따른 값** (시험용, 두 묶음을 합친 문항. 식별기는 모두 v11이고 문장만 다르다. 괄호 안은 경주 단위 95% 구간)\n\n" + "\n".join(t1)
         + "\n\n**검사가 한 일**\n\n" + "\n".join(t2))
(HERE / ("results_llm.json" if a.split == "test" else f"results_llm_{a.split}.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
readme = (HERE / "README_KO.md").read_text()
if "<!-- results-llm -->" in readme and a.split == "test" and not a.no_readme:
    (HERE / "README_KO.md").write_text(re.sub(r"<!-- results-llm -->.*?<!-- /results-llm -->", lambda m: "<!-- results-llm -->\n" + table + "\n<!-- /results-llm -->", readme, flags=re.S))
print(table)
