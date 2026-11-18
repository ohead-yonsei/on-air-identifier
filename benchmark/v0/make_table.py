"""ON-AIR draft benchmark v0: score every baseline on the test split and write the table into README_KO.md and results.json.
usage: python3 benchmark/v0/make_table.py        (standard library only)

A baseline whose run is not finished is scored on the items it has (its items are a random sample: baseline_vlm.py shuffles
with a fixed seed), and the table says how many."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score

ROWS = [("confirmed_only", "번호를 말하지 않기 (확정 정보만)"), ("random_three", "출전 번호 셋을 무작위로 말하기"),
        ("vlm_qwen9b_free", "화면만 본 모델 (Qwen 3.5 9B, \"누가 앞서 있는지 중계하라\")"),
        ("vlm_qwen9b_careful", "화면만 본 모델 (같은 모델, \"확실할 때만 번호를 말하라\")"),
        ("ours_identifier_F2", "**우리 흐름** (식별기 F2 + 정해진 문장 틀)")]
pc = lambda v: "—" if v is None else f"{v:.1f}%"
res, lines = {}, ["| 기준선 | 문항 | 틀린 문장 비율 ↓ | A 번호 정확도 ↑ | B 말한 양 ↑ | 선두 주장 (맞음 / 말함) |", "|---|---|---|---|---|---|"]
for name, label in ROWS:
    f = HERE / "baselines" / f"{name}.jsonl"
    if not f.exists():
        continue
    rs = [r for r in score.rows(f, "test") if r["answered"]]
    s = score.summarise(rs)
    s["interval"] = {k: score.interval(rs, k) for k in ("hallucination", "A", "B", "leader")}
    res[name] = s
    iv = lambda k: "" if s["interval"][k] is None else f" ({s['interval'][k][0]:.0f}~{s['interval'][k][1]:.0f})"
    lead = "—" if not s["leader_claims"] else f"{pc(s['leader'])} ({s['leader_right']} / {s['leader_claims']})"
    lines.append(f"| {label} | {s['items']} | {pc(s['hallucination'])}{iv('hallucination')} | {pc(s['A'])}{iv('A')} ({s['numbers_right']} / {s['numbers']}) | {pc(s['B'])}{iv('B')} | {lead} |")
table = "\n".join(lines) + "\n\n괄호 안의 범위는 경주 단위로 다시 뽑은 95% 구간이다. 문항 수가 874보다 적은 줄은 아직 다 돌리지 못해 무작위로 뽑힌 일부만 채점한 것이다."
(HERE / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
readme = (HERE / "README_KO.md").read_text()
if "RESULTS_TABLE" in readme:
    readme = readme.replace("RESULTS_TABLE", "<!-- results -->\n" + table + "\n<!-- /results -->")
else:
    readme = re.sub(r"<!-- results -->.*?<!-- /results -->", lambda m: "<!-- results -->\n" + table + "\n<!-- /results -->", readme, flags=re.S)
(HERE / "README_KO.md").write_text(readme)
print(table)
