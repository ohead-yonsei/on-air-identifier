"""Commentary demo of one day as a readable document: our flow next to the picture-only model, with the order at that moment.
usage: python3 code/cl9_view.py DAY        (standard library only; DAY = 20261115)

Reads live_commentary/DAY/commentary_v3 (identifier L1 with leader and order -> Qwen writes the sentence) and commentary_v2
(for the sentence the same model wrote from the picture alone), the strip stored with each time point, and the official passing
times (finish2/official, finish2/times). Writes live_commentary/DAY/COMMENTARY_KO.md.
A stated number counts as wrong when it is outside the strip's first three (strip on screen) or, without a strip, outside the
first three computed from the official times (within 3 m of the third still counts as right). Added after the fact, for reading."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn_common as fc

day = sys.argv[1]
O = fc.ROOT / "live_commentary" / day
off, tms = fc.official(day), fc.times(day)
out, tot = [], {"ours": [0, 0, 0, 0], "pic": [0, 0, 0, 0]}        # [strip sentences, wrong among them, no-strip sentences, wrong among them]
body = []
for f in sorted((O / "commentary_v3").glob("R??.json")):
    race = f.stem
    a, b = json.loads(f.read_text()), json.loads((O / "commentary_v2" / f.name).read_text())
    pic = {r["k"]: r for r in b["rows"]}
    tw, S = fc.winner_time(off[race]), tms[race].get("S", a["info"]["start_video_time"])
    top = [h["number"] for h in off[race]["horses"] if h["rank"] in (1, 2, 3)]
    body.append(f"\n## {a['info']['race']}경주 ({a['info']['distance']}m)\n\n공식 결과 1~3위: {', '.join(f'{n}번' for n in top)}. 우승 기록 {int(tw // 60)}:{tw % 60:04.1f}.\n")
    body.append("| 경과 | 구간 | 우리 흐름 (식별기 + 문장 모델) | 화면만 본 모델 | 그 순간 1~3위 |")
    body.append("|---|---|---|---|---|")
    for r in a["rows"]:
        rt = r["t"] - S + 0.1
        ref = fc.reference_full(off[race], rt) if 0 < rt <= tw + 0.3 else None
        strip = r.get("strip")
        if strip:
            good, where = set(strip[:3]), "띠 " + "-".join(map(str, strip[:3]))
        elif ref:
            good, where = set(ref["top3_lenient"]), "공식 " + "-".join(map(str, ref["top3"]))
        else:
            good, where = None, "결승 뒤" if rt > tw else "—"
        cells = []
        for key, line, nums in (("ours", r["line"], r["said_numbers"]), ("pic", pic[r["k"]]["picture_only_line"], pic[r["k"]]["picture_only_numbers"])):
            wrong = [n for n in nums if good is not None and n not in good]
            if good is not None:
                i = 0 if strip else 2
                tot[key][i] += 1
                tot[key][i + 1] += bool(wrong)
            cells.append((line or "(문장 없음)").replace("|", "/") + (f" **← 틀린 번호 {', '.join(map(str, wrong))}**" if wrong else ""))
        mode = "순위" if r["mode"] == "leader" else "확정 정보"
        body.append(f"| {int(round(r['elapsed']))}초 | {mode} | {cells[0]} | {cells[1]} | {where} |")
o, p = tot["ours"], tot["pic"]
head = f"""# 하루치 중계 시연 — 문장 전체 ({day[:4]}-{day[4:6]}-{day[6:]} 서울 1~11경주)

3초마다 한 문장. 띠는 어느 쪽도 보지 않았다.

- **우리 흐름:** 식별기(L1, 선두와 순서 포함)가 낸 번호를 로컬 Qwen 9B에 주고 문장을 쓰게 한 것(세 번째 실행). 식별기가 확실하지 않거나 출발 직후·결승 30초 전부터는 번호 없이 확정 정보로 말한다.
- **화면만 본 모델:** 같은 Qwen 9B가 그 순간의 화면만 보고 쓴 문장.
- **그 순간 1~3위:** 띠가 화면에 있으면 띠의 1~3위, 없으면 공식 경주 기록으로 계산한 1~3위.
- **틀린 번호:** 문장에 나온 번호가 그 1~3위 밖이면 틀린 번호로 표시했다(띠가 없을 때는 3위와 3m 안이면 맞는 것으로 봄).

| 틀린 번호가 든 문장 | 띠가 있던 시점 | 띠가 없던 시점 |
|---|---|---|
| 우리 흐름 | {o[0]}개 중 {o[1]}개 | {o[2]}개 중 {o[3]}개 |
| 화면만 본 모델 | {p[0]}개 중 {p[1]}개 | {p[2]}개 중 {p[3]}개 |

띠가 없던 시점의 채점은 공식 기록 정답이 생긴 뒤에 덧붙인 것이다(참고값). 결승 뒤의 문장은 세지 않았다.
"""
(O / "COMMENTARY_KO.md").write_text(head + "\n".join(body) + "\n", encoding="utf-8")
print(head)
