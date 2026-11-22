"""score1: every stored Korean baseline scored with rules v0 and v1, side by side (test split, both tracks).
usage: sc2_rescore.py OUT.json      standard library only"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "benchmark" / "v0"
sys.path.insert(0, str(B))
import score

FOUR = ("20260905_R09", "20260912_R08", "20260919_R04", "20261003_R08")
ROWS = [("번호를 말하지 않기", ["confirmed_only", "confirmed_only_official"], None), ("무작위 셋", ["random_three", "random_three_official"], None),
        ("로컬 Qwen 3.5 9B, 자유롭게", ["vlm_qwen9b_free", "vlm_qwen9b_free_official"], None),
        ("로컬 Qwen 3.5 9B, 확실할 때만", ["vlm_qwen9b_careful", "vlm_qwen9b_careful_official"], None),
        ("식별기 v9 + 문장 틀", ["ours_identifier_F2", "ours_identifier_F2_official"], None),
        ("식별기 v11 + 문장 틀", ["ours_identifier_v11", "ours_identifier_v11_official"], None),
        ("식별기 v11 + 우승마 부르기", ["ours_identifier_v11w", "ours_identifier_v11w_official"], None),
        ("v11 + 모델이 쓴 문장 (검사 없음)", ["ours_v11_llm_ko_raw", "ours_v11_llm_ko_official_raw"], None),
        ("v11 + 모델이 쓴 문장 + 검사", ["ours_v11_llm_ko", "ours_v11_llm_ko_official"], None),
        ("(4개 경주) 식별기 v11 + 문장 틀", ["ours_identifier_v11", "ours_identifier_v11_official"], FOUR),
        ("(4개 경주) Terra, 자유롭게", ["vlm_terra_free_4races"], FOUR), ("(4개 경주) Terra, 확실할 때만", ["vlm_terra_careful_4races"], FOUR)]
res = {}
for label, files, races in ROWS:
    lines = [l for f in files for l in (B / "baselines" / f"{f}.jsonl").read_text().splitlines() if l and not json.loads(l).get("error")]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines) + "\n")
    res[label] = {}
    for rules in ("v0", "v1"):
        score.RULES = rules
        rs = [r for r in score.rows(f.name, "test", "all") if r["answered"] and (races is None or r["race"] in races)]
        s = score.summarise(rs)
        res[label][rules] = {"items": s["items"], "leader right": s["leader_right_share"], "leader wrong": s["leader_wrong_share"],
                             "no leader said": round(100 - s["leader_right_share"] - s["leader_wrong_share"], 1),
                             "items right": s["items_leader_right"], "items wrong": s["items_leader_wrong"]}
    Path(f.name).unlink()
score.RULES = "v0"
Path(sys.argv[1]).write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(f"{'row':36s} items | v0: right wrong none | v1: right wrong none")
for label, r in res.items():
    a, b = r["v0"], r["v1"]
    print(f"{label:36s} {a['items']:5} | {a['leader right']:5} {a['leader wrong']:5} {a['no leader said']:5} | {b['leader right']:5} {b['leader wrong']:5} {b['no leader said']:5}")
