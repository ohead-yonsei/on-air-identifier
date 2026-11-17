"""Build one self-contained HTML page that shows, time point by time point, what identify_l1.py said on a video.
usage: python3 demo2_page.py DEMO_DIR      (DEMO_DIR holds overlays/summary.json, overlays/*.jpg, finish.json; writes DEMO_DIR/demo.html)
Standard library only, so it runs on the Mac."""
import base64
import html
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
S = json.loads((D / "overlays" / "summary.json").read_text())
F = json.loads((D / "finish.json").read_text())
top3, finish_t = F["order"][:3], F["finish_time"]
score = S["score_on_strip_points"]


def chip(n, cls, note):
    return f'<span class="chip {cls}"><b>{n}번</b> {html.escape(note)}</span>'


cards, late = [], {"points": 0, "statements": 0, "in_top3": 0, "silent": 0}
for p in S["points"]:
    img = base64.b64encode((D / "overlays" / p["image"]).read_bytes()).decode()
    chips = []
    if p["strip_on_screen"]:
        for n in p["statements"]:
            if n in p["right"]:
                chips.append(chip(n, "ok", "맞음 (띠 상위 3두)"))
            elif n in p["in_places_4_5"]:
                chips.append(chip(n, "warn", "띠 4~5위"))
            else:
                chips.append(chip(n, "bad", "띠에 없음"))
        zone = '<span class="zone on">순위 띠 있음 · 확인된 구간</span>'
        ref = "띠: " + " · ".join(f"{i + 1}위 {n}번" for i, n in enumerate(p["strip"]))
    else:
        after = p["t"] > 40
        for n in p["statements"]:
            chips.append(chip(n, "refok" if n in top3 else "refno", "결승 1~3위" if n in top3 else "결승 1~3위 아님"))
        zone = '<span class="zone off">순위 띠 없음 · 확인되지 않은 구간</span>'
        ref = "띠가 화면에 없다. 참고 기준은 결승 순서(" + ", ".join(f"{n}번" for n in top3) + ")"
        if after:
            late["points"] += 1
            late["statements"] += len(p["statements"])
            late["in_top3"] += sum(n in top3 for n in p["statements"])
            late["silent"] += not p["statements"]
    if not chips:
        chips.append('<span class="chip none">말하지 않음</span>')
    det = []
    for d in p["detail"]:
        if d["source"] == "now":
            det.append(f'{d["number"]}번: 세 장 중 {d["votes"]}장, 모자 확률 {d["p_cap"]:.2f}, 상위 3두 확률 {d["p_lead"]:.2f}')
        else:
            det.append(f'{d["number"]}번: 3초 전과 6초 전에 말한 번호를 이어받음')
    small = "작은 기수 검출 사용" if "D3" in p["detections"] else "보통 검출"
    left = finish_t - p["t"]
    cards.append(f'''<article class="card">
<img src="data:image/jpeg;base64,{img}" alt="영상 {p["t"]:.0f}초 화면에 식별기 결과를 그린 그림" loading="lazy">
<div class="body">
<div class="when"><b>영상 {p["t"]:.0f}초</b> <span>결승 {left:.0f}초 전</span> {zone}</div>
<div class="said">{"".join(chips)}</div>
<p class="ref">{html.escape(ref)}</p>
<p class="det">{html.escape(" / ".join(det)) if det else "두 장 이상에서 고른 번호가 없다"}</p>
<p class="meta">기수 상자 {p["riders"]}개, 번호 확정 {p["sure"]}개 · {small}</p>
</div></article>''')

n_points = len(S["points"])
page = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>식별기 예시 — 2026-11-15 서울 1경주</title>
<style>
:root {{ --bg:#f7f6f3; --card:#ffffff; --ink:#1c1b19; --quiet:#6b6860; --line:#dedbd3; --ok:#1f7a3d; --okbg:#e3f3e7; --warn:#8a5a00; --warnbg:#fbefd2; --bad:#a3271f; --badbg:#fbe3e0; --ref:#1f5fa8; --refbg:#e3eefb; --off:#5f5c55; --offbg:#eceae4; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#171614; --card:#22211e; --ink:#eceae4; --quiet:#a29f96; --line:#3a3833; --ok:#7fd49a; --okbg:#1d3324; --warn:#e9c46a; --warnbg:#3a3015; --bad:#f09a92; --badbg:#3d1f1c; --ref:#8fbcf0; --refbg:#1b2c40; --off:#b5b2a9; --offbg:#302e2a; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:28px 16px 60px; }}
h1 {{ font-size:24px; line-height:1.3; margin:0 0 6px; }}
h2 {{ font-size:17px; margin:30px 0 10px; }}
.lead {{ color:var(--quiet); margin:0 0 18px; }}
table {{ border-collapse:collapse; width:100%; max-width:760px; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ color:var(--quiet); font-weight:600; font-size:13px; }}
ul {{ margin:6px 0; padding-left:20px; }} li {{ margin:3px 0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
.card img {{ display:block; width:100%; height:auto; }}
.body {{ padding:12px 14px 14px; }}
.when {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:8px; }}
.when span {{ color:var(--quiet); font-size:13px; }}
.zone {{ font-size:12px !important; padding:2px 8px; border-radius:999px; }}
.zone.on {{ background:var(--okbg); color:var(--ok) !important; }} .zone.off {{ background:var(--offbg); color:var(--off) !important; }}
.said {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }}
.chip {{ font-size:13px; padding:3px 9px; border-radius:6px; }}
.chip.ok {{ background:var(--okbg); color:var(--ok); }} .chip.warn {{ background:var(--warnbg); color:var(--warn); }} .chip.bad {{ background:var(--badbg); color:var(--bad); }}
.chip.refok {{ background:var(--refbg); color:var(--ref); }} .chip.refno {{ background:var(--warnbg); color:var(--warn); }} .chip.none {{ background:var(--offbg); color:var(--off); }}
.ref, .det, .meta {{ margin:4px 0 0; font-size:13px; color:var(--quiet); }}
.det {{ color:var(--ink); }}
.key span {{ display:inline-block; width:14px; height:14px; border-radius:3px; vertical-align:-2px; margin-right:6px; }}
</style></head><body><main>
<h1>식별기 예시: 새 영상 한 편을 처음부터 끝까지</h1>
<p class="lead">2026-11-15 서울 1경주(1000m). 학습에도 시험에도 쓴 적 없는 영상이다. 식별기 L1을 그대로 돌렸고, 3초마다 한 번씩 {n_points}개 시점이 나왔다.</p>

<h2>요약</h2>
<table>
<tr><th>구간</th><th>시점</th><th>말한 번호</th><th>결과</th></tr>
<tr><td>순위 띠가 화면에 있는 구간</td><td>{score["validated_points"]}개</td><td>{score["statements"]}개</td><td>{score["right"]}개가 띠의 상위 3두. 모든 시점에서 맞는 번호를 하나 이상 말했다</td></tr>
<tr><td>띠가 사라진 뒤 (마지막 {finish_t - 48:.0f}초쯤)</td><td>{late["points"]}개</td><td>{late["statements"]}개</td><td>{late["in_top3"]}개가 결승 1~3위였다. 마지막 {late["silent"]}개 시점에서는 말하지 않았다</td></tr>
</table>
<ul>
<li>띠가 사라진 뒤에도 식별기는 4번과 8번을 계속 말했다. 실제 결승은 8번, 7번, 4번 순이었다. 2위로 들어온 7번(갈색 모자)은 한 번도 말하지 않았다.</li>
<li>결승 직전 두 시점에서는 아무 번호도 말하지 않았다. 화면이 멀어져 번호를 확정하지 못했기 때문이다.</li>
<li>띠가 없는 구간의 판정은 참고용이다. 기준이 "결승선을 지난 순서"라서 그 순간의 선두권과 다를 수 있다. 예를 들어 영상 52초의 5번은 6초 전 띠에서 2위였다.</li>
</ul>

<h2>그림 읽는 법</h2>
<ul class="key">
<li><span style="background:#1e78eb"></span>파란 상자: 이 화면에서 "앞선 말"로 고른 기수</li>
<li><span style="background:#3caa3c"></span>초록 상자: 번호는 확정했지만 앞선 말로 고르지 않은 기수</li>
<li><span style="background:#969696"></span>회색 상자(번호 뒤 물음표): 번호를 확정하지 못한 기수</li>
<li>그림은 세 장 가운데 가장 최근 한 장이다. 말하는 번호는 세 장 중 두 장 이상에서 고른 것과, 3초·6초 전에 둘 다 말한 것이다. 그래서 파란 상자와 말한 번호가 다를 수 있다.</li>
<li>화면 아래의 순위 띠와 타이머는 보기 편하게 남겨 둔 것이다. 식별기는 이 그래픽을 가린 화면을 받았다.</li>
</ul>

<h2>시점별 결과</h2>
<div class="grid">
{"".join(cards)}
</div>

<h2>이 예시의 한계</h2>
<ul>
<li>경주 하나다. 숫자가 좋게 나온 것은 예시일 뿐이고, 성능은 여러 경주일로 잰 값(말한 것 중 92.9% 정답, 94.6%의 시점에서 맞는 말)을 봐야 한다.</li>
<li>결승 순서는 경주 뒤의 느린 재생 화면에서 Claude가 번호판을 읽어 적었다. 사람이 검수하지 않았다.</li>
<li>처리에 {S["seconds"]:.0f}초가 걸렸다(시점 하나에 {S["seconds"] / n_points:.0f}초쯤). Qwen 호출 실패는 {S["qwen_failures"]}회다.</li>
</ul>
</main></body></html>'''
(D / "demo.html").write_text(page, encoding="utf-8")
print("wrote", D / "demo.html", round(len(page) / 1e6, 2), "MB | late:", late)
