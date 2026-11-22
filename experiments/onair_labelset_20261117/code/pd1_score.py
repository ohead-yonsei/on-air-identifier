"""paid1: score the paid picture-only baseline on the four drawn test races, next to the stored baselines on the same items.
usage: python3 pd1_score.py OUT.json NAME [NAME ...]        NAME = a file of baseline_openai.py in benchmark/v0/baselines (both tracks in one file)
Standard library only. Lines with an error are left out (the item then counts as not answered and is not scored for that row)."""
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "benchmark" / "v0"
sys.path.insert(0, str(B))
import score

RACES = ("20260905_R09", "20260912_R08", "20260919_R04", "20261003_R08")
STORED = [("confirmed_only", "번호를 말하지 않기"), ("random_three", "무작위 셋"), ("vlm_qwen9b_free", "로컬 Qwen 3.5 9B, 자유롭게"),
          ("vlm_qwen9b_careful", "로컬 Qwen 3.5 9B, 확실할 때만"), ("ours_identifier_F2", "식별기 v9 + 문장 틀"), ("ours_identifier_v11", "식별기 v11 + 문장 틀")]
out_path, names = Path(sys.argv[1]), sys.argv[2:]


def measure(lines):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + "\n")
    res = {}
    for track in ("all", "strip", "official"):
        rs = [r for r in score.rows(f.name, "test", track) if r["race"] in RACES and r["answered"]]
        s = score.summarise(rs)
        n = s["items"]
        res[track] = {"items": n, "leader right": s["leader_right_share"], "leader wrong": s["leader_wrong_share"],
                      "no leader said": round(100 - (s["leader_right_share"] or 0) - (s["leader_wrong_share"] or 0), 1) if n else None,
                      "leader wrong, within 3 m counted right": s["leader_wrong_share_lenient"] if track == "official" else None,
                      "sentences with a wrong number": s["hallucination"], "A": s["A"], "B": s["B"],
                      "numbers per item": round(s["numbers"] / n, 2) if n else None, "numbers not in the race": s["numbers_not_in_race"]}
    Path(f.name).unlink()
    return res


res = {"races": RACES, "rows": {}}
for name, label in STORED:
    lines = [json.loads(l) for suf in ("", "_official") for l in (B / "baselines" / f"{name}{suf}.jsonl").read_text().splitlines() if l]
    res["rows"][label] = measure(lines)
for name in names:
    raw = [json.loads(l) for l in (B / "baselines" / f"{name}.jsonl").read_text().splitlines() if l]
    good = {r["id"]: r for r in raw if not r.get("error")}
    m = measure(list(good.values()))
    ms = sorted(r["ms"] for r in good.values())
    m["calls"] = {"answered": len(good), "lines with an error": sum(1 for r in raw if r.get("error")), "seconds per call, median": round(statistics.median(ms) / 1000, 2) if ms else None,
                  "seconds per call, 90th percentile": round(ms[int(0.9 * (len(ms) - 1))] / 1000, 2) if ms else None,
                  "tokens in per call": round(statistics.mean(r["tokens_in"] or 0 for r in good.values())) if good else None,
                  "tokens out per call": round(statistics.mean(r["tokens_out"] or 0 for r in good.values())) if good else None,
                  "dollars in all (from counted tokens)": round(sum(r.get("usd", 0) for r in raw), 3),
                  "dollars per race": round(sum(r.get("usd", 0) for r in raw) / len(RACES), 3)}
    res["rows"][name] = m
out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
for track in ("all", "strip", "official"):
    print(f"\n== {track}")
    print(f"   {'row':34s} items  right  wrong  silent  wrong(3m)  wrong-sentence     A      B  n/item  not-in-race")
    for label, m in res["rows"].items():
        t = m[track]
        print(f"   {label:34s} {t['items']:5} {t['leader right']!s:>6} {t['leader wrong']!s:>6} {t['no leader said']!s:>7} {t['leader wrong, within 3 m counted right']!s:>10} "
              f"{t['sentences with a wrong number']!s:>14} {t['A']!s:>6} {t['B']!s:>6} {t['numbers per item']!s:>7} {t['numbers not in the race']!s:>12}")
for name in names:
    print(name, res["rows"][name]["calls"])
