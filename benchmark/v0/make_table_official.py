"""ON-AIR draft benchmark v0, official track: score every baseline on the test split, by part of the race, and write the tables into
README_KO.md (between the results-official marks) and results_official.json.
usage: python3 benchmark/v0/make_table_official.py        (standard library only)

Parts: start = race time under 18 s, finish = the last 36 s (the finish itself included), finish itself = the time point at the finish.
The few items in the middle of the race (moments the strip was not read) are in "all" only.
Also scored: both tracks together, for the baselines that have both files, and for the v11 flow the three test days that gave no mined
cap labels (2026-09-12, 09-05, 08-30)."""
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
        ("ours_identifier_F2", "우리 흐름 (식별기 v9 + 정해진 문장 틀)"),
        ("ours_identifier_v11", "**우리 흐름** (식별기 v11 + 같은 문장 틀)"),
        ("ours_identifier_v11w", "**우리 흐름** (v11 + 결승 순간에 우승마 부르기)")]
PARTS = [(None, "전체"), ("start", "출발 직후"), ("finish", "마지막 36초"), ("finish_itself", "결승 순간")]
CLEAN = ("20260912", "20260905", "20260830")
pc = lambda v: "—" if v is None else f"{v:.1f}%"


def measure(rs):
    s = score.summarise(rs)
    s["interval"] = {k: score.interval(rs, k) for k in score.KEYS}
    return s


def concat(name):
    """One predictions file for both tracks (ids do not overlap)."""
    f = HERE / "baselines" / f".both_{name}.jsonl"
    f.write_text((HERE / "baselines" / f"{name}.jsonl").read_text() + (HERE / "baselines" / f"{name}_official.jsonl").read_text())
    return f


res = {}
t1 = ["| 기준선 | " + " | ".join(f"{label}<br>맞게 ↑ / 틀리게 ↓" for _, label in PARTS) + " |", "|---|" + "---|" * len(PARTS)]
t2 = ["| 기준선 | 문항 | 선두를 틀리게 (넉넉한 정답) ↓ | 틀린 문장 비율 ↓ (넉넉한 정답) | A ↑ (넉넉한 정답) | B ↑ |", "|---|---|---|---|---|---|"]
t3 = ["| 기준선 | 문항 | 선두를 맞게 말함 ↑ | 선두를 틀리게 말함 ↓ | 선두를 말하지 않음 | 틀린 문장 비율 ↓ | A ↑ | B ↑ |", "|---|---|---|---|---|---|---|---|"]
counts = None
for name, label in ROWS:
    f = HERE / "baselines" / f"{name}_official.jsonl"
    if not f.exists():
        continue
    by = {}
    for part, _ in PARTS:
        by[part or "all"] = measure([r for r in score.rows(f, "test", "official", part) if r["answered"]])
    res[name] = by
    counts = counts or {k: v["items"] for k, v in by.items()}
    cell = lambda s: "—" if not s["items"] else f"{pc(s['leader_right_share'])} / {pc(s['leader_wrong_share'])}"
    t1.append(f"| {label} | " + " | ".join(cell(by[p or "all"]) for p, _ in PARTS) + " |")
    s = by["all"]
    iv = lambda k: "" if s["interval"][k] is None else f" ({s['interval'][k][0]:.0f}~{s['interval'][k][1]:.0f})"
    t2.append(f"| {label} | {s['items']} | {pc(s['leader_wrong_share'])}{iv('leader_wrong_share')} ({pc(s['leader_wrong_share_lenient'])}) | {pc(s['hallucination'])}{iv('hallucination')} ({pc(s['hallucination_lenient'])}) "
              f"| {pc(s['A'])}{iv('A')} ({pc(s['A_lenient'])}) | {pc(s['B'])}{iv('B')} |")
    if (HERE / "baselines" / f"{name}.jsonl").exists():
        both = concat(name)
        rs = [r for r in score.rows(both, "test", "all") if r["answered"]]
        s = measure(rs)
        res[name]["both tracks"] = s
        iv = lambda k: "" if s["interval"][k] is None else f" ({s['interval'][k][0]:.0f}~{s['interval'][k][1]:.0f})"
        none = s["items"] - s["items_leader_right"] - s["items_leader_wrong"]
        t3.append(f"| {label} | {s['items']} | {pc(s['leader_right_share'])}{iv('leader_right_share')} | {pc(s['leader_wrong_share'])}{iv('leader_wrong_share')} | {pc(100 * none / s['items'])} "
                  f"| {pc(s['hallucination'])}{iv('hallucination')} | {pc(s['A'])}{iv('A')} | {pc(s['B'])}{iv('B')} |")
        if name.startswith("ours"):
            for track in ("official", "all"):
                src = f if track == "official" else both
                res[name][f"clean three days, {track}"] = measure([r for r in score.rows(src, "test", track) if r["answered"] and r["race"][:8] in CLEAN])
        both.unlink()
n = counts or {}
table = (f"**구간별 대표 지표** (시험용, 띠 없는 문항 {n.get('all')}개: 출발 직후 {n.get('start')}, 마지막 36초 {n.get('finish')}, 그 가운데 결승 순간 {n.get('finish_itself')}. 칸의 값은 선두를 맞게 말한 문항 / 틀리게 말한 문항의 비율)\n\n"
         + "\n".join(t1) + "\n\n**띠 없는 문항 전체의 다른 지표** (괄호 안 범위는 경주 단위 95% 구간, 그 뒤 괄호는 넉넉한 정답으로 센 값)\n\n" + "\n".join(t2)
         + "\n\n**두 묶음을 합친 값** (띠가 있는 문항 874개 + 띠 없는 문항)\n\n" + "\n".join(t3))
(HERE / "results_official.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
readme = (HERE / "README_KO.md").read_text()
if "<!-- results-official -->" in readme:
    readme = re.sub(r"<!-- results-official -->.*?<!-- /results-official -->", lambda m: "<!-- results-official -->\n" + table + "\n<!-- /results-official -->", readme, flags=re.S)
    (HERE / "README_KO.md").write_text(readme)
print(table)
