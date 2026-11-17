"""AI commentary demo, step 4: one self-contained HTML page with the commentary of every race, time point by time point.
usage: python3 cl4_page.py DAY_DIR     (DAY_DIR = live_commentary/20261115; needs commentary/*.json, score.json, thumbs/Rxx/overlay_NNN.jpg)
Standard library only, so it runs on the Mac."""
import base64
import html
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
SUB = sys.argv[2] if len(sys.argv) > 2 else ""      # "_v2" for the second run
score = json.loads((D / f"score{SUB}.json").read_text())
S, P = score["total"]["system"], score["total"]["picture_only"]


def chips(nums, strip, scored):
    out = []
    for n in nums:
        if not scored:
            out.append(f'<span class="chip none">{n}번 기준 없음</span>')
        elif n in strip[:3]:
            out.append(f'<span class="chip ok">{n}번 맞음</span>')
        elif n in strip:
            out.append(f'<span class="chip warn">{n}번 띠 4~5위</span>')
        else:
            out.append(f'<span class="chip bad">{n}번 틀림</span>')
    return "".join(out)


sections = []
for f in sorted((D / f"commentary{SUB}").glob("R*.json")):
    d = json.loads(f.read_text())
    race, info = d["race"], d["info"]
    per = score["by_race"][race]
    rows = []
    for r in d["rows"]:
        tp = D / "thumbs" / race / f"overlay_{r['k']:03d}.jpg"
        img = f'<img src="data:image/jpeg;base64,{base64.b64encode(tp.read_bytes()).decode()}" alt="출발 뒤 {r["elapsed"]:.0f}초 화면" loading="lazy">' if tp.exists() else ""
        strip = r["strip"]
        mode = '<span class="mode lead">선두 중계</span>' if r["mode"] == "leader" else '<span class="mode conf">확정 정보</span>'
        note = {"model": "", "model_retry": " · 다시 만든 문장", "template": " · 문장 틀"}[r["source"]]
        sys_chips = chips(r["said_numbers"], strip, bool(strip)) if r["mode"] == "leader" else ("".join(f'<span class="chip none">{n}번 과거형 언급</span>' for n in r["said_numbers"]))
        ref = ("띠: " + ", ".join(f"{n}번" for n in strip[:3]) + " (4~5위 " + ", ".join(f"{n}번" for n in strip[3:]) + ")") if strip else "띠 없음"
        rows.append(f'''<tr>
<td class="pic">{img}</td>
<td class="t"><b>{r["elapsed"]:.0f}초</b><br><span>{html.escape(r["phase"])}</span><br>{mode}</td>
<td class="line"><p>{html.escape(r["line"])}</p><div>{sys_chips}</div><small>식별기: {", ".join(f"{n}번" for n in r["identifier"]) or "말하지 않음"}{note}</small></td>
<td class="line base"><p>{html.escape(r["picture_only_line"])}</p><div>{chips(r["picture_only_numbers"], strip, bool(strip))}</div></td>
<td class="ref">{html.escape(ref)}</td>
</tr>''')
    s, p = per["system"], per["picture_only"]
    sections.append(f'''<section id="{race}">
<h2>서울 {info["race"]}경주 · {info["distance"]}m</h2>
<p class="sub">시점 {per["points"]}개 가운데 선두 중계 {per["leader_points"]}개. 띠로 채점한 번호: 식별기 중계 {s["right"]}/{s["numbers"]} 맞음, 그림만 본 모델 {p["right"]}/{p["numbers"]} 맞음.</p>
<table class="tl"><thead><tr><th>화면</th><th>출발 뒤</th><th>식별기를 쓴 중계</th><th>그림만 본 모델의 중계</th><th>채점 기준 (시스템은 못 봄)</th></tr></thead><tbody>
{"".join(rows)}
</tbody></table></section>''')


def pct(a, b):
    return f"{a / b * 100:.1f}%" if b else "—"


nav = " ".join(f'<a href="#{r}">{int(r[1:])}경주</a>' for r in sorted(score["by_race"]))
page = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI 중계 시연 — 2026-11-15 서울</title>
<style>
:root {{ --bg:#f7f6f3; --card:#ffffff; --ink:#1c1b19; --quiet:#6b6860; --line:#dedbd3; --ok:#1f7a3d; --okbg:#e3f3e7; --warn:#8a5a00; --warnbg:#fbefd2; --bad:#a3271f; --badbg:#fbe3e0; --off:#5f5c55; --offbg:#eceae4; --lead:#1f5fa8; --leadbg:#e3eefb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#171614; --card:#22211e; --ink:#eceae4; --quiet:#a29f96; --line:#3a3833; --ok:#7fd49a; --okbg:#1d3324; --warn:#e9c46a; --warnbg:#3a3015; --bad:#f09a92; --badbg:#3d1f1c; --off:#b5b2a9; --offbg:#302e2a; --lead:#8fbcf0; --leadbg:#1b2c40; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; }}
main {{ max-width:1240px; margin:0 auto; padding:28px 16px 60px; }}
h1 {{ font-size:24px; line-height:1.3; margin:0 0 6px; }}
h2 {{ font-size:18px; margin:34px 0 4px; }}
.lead, .sub {{ color:var(--quiet); margin:0 0 14px; }}
nav {{ margin:14px 0; display:flex; flex-wrap:wrap; gap:8px; }}
nav a {{ color:var(--lead); text-decoration:none; padding:3px 10px; border:1px solid var(--line); border-radius:999px; font-size:13px; }}
table {{ border-collapse:collapse; width:100%; }}
table.sum {{ max-width:860px; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ color:var(--quiet); font-weight:600; font-size:13px; }}
.wrap {{ overflow-x:auto; }}
table.tl {{ background:var(--card); border:1px solid var(--line); border-radius:10px; min-width:980px; }}
td.pic {{ width:210px; }} td.pic img {{ display:block; width:200px; height:auto; border-radius:6px; }}
td.t {{ width:96px; white-space:nowrap; }} td.t span {{ color:var(--quiet); font-size:12.5px; }}
td.line p {{ margin:0 0 6px; font-size:15.5px; }} td.line.base p {{ color:var(--quiet); }}
td.line small {{ color:var(--quiet); display:block; margin-top:4px; }}
td.ref {{ width:150px; color:var(--quiet); font-size:12.5px; }}
.mode {{ font-size:12px; padding:2px 8px; border-radius:999px; }}
.mode.lead {{ background:var(--leadbg); color:var(--lead); }} .mode.conf {{ background:var(--offbg); color:var(--off); }}
.chip {{ display:inline-block; font-size:12.5px; padding:2px 8px; border-radius:6px; margin:0 4px 4px 0; }}
.chip.ok {{ background:var(--okbg); color:var(--ok); }} .chip.warn {{ background:var(--warnbg); color:var(--warn); }} .chip.bad {{ background:var(--badbg); color:var(--bad); }} .chip.none {{ background:var(--offbg); color:var(--off); }}
ul {{ margin:6px 0; padding-left:20px; }} li {{ margin:3px 0; }}
</style></head><body><main>
<h1>식별기를 써서 실제로 중계해 보기</h1>
<p class="lead">2026-11-15 서울 {score["races"]}개 경주, 3초마다 한 문장씩 {score["time_points"]}개 시점. 중계 시스템은 순위 띠를 보지 않았다. 띠는 채점할 때만 썼다.</p>

<h2>요약</h2>
<div class="wrap"><table class="sum">
<tr><th>띠가 화면에 있던 시점에서</th><th>식별기를 쓴 중계</th><th>그림만 본 모델의 중계</th></tr>
<tr><td>말한 번호</td><td>{S["numbers"]}개</td><td>{P["numbers"]}개</td></tr>
<tr><td>그중 띠의 상위 3두 (맞음)</td><td><b>{S["right"]}개 ({pct(S["right"], S["numbers"])})</b></td><td>{P["right"]}개 ({pct(P["right"], P["numbers"])})</td></tr>
<tr><td>띠 4~5위</td><td>{S["places_4_5"]}개</td><td>{P["places_4_5"]}개</td></tr>
<tr><td>띠에 없는 번호</td><td>{S["off_strip"]}개</td><td>{P["off_strip"]}개</td></tr>
<tr><td>틀린 번호가 들어간 문장</td><td>{S["points_with_wrong"]}개 / {S["points_with_strip"]}개 시점</td><td>{P["points_with_wrong"]}개 / {P["points_with_strip"]}개 시점</td></tr>
<tr><td>맞는 번호가 들어간 문장</td><td>{S["points_with_right"]}개 / {S["points_with_strip"]}개 시점</td><td>{P["points_with_right"]}개 / {P["points_with_strip"]}개 시점</td></tr>
<tr><td>띠가 없던 시점에 말한 번호 (기준 없음)</td><td>{S["no_reference"]}개</td><td>{P["no_reference"]}개</td></tr>
</table></div>
<ul>
<li>선두 중계 모드는 {score["modes"]["leader"]}개 시점, 확정 정보 모드는 {score["modes"]["confirmed"]}개 시점이다. 확정 정보 모드에서는 지금 누가 앞서는지 말하지 않는다.</li>
<li>코드가 문장을 검사해 다시 만들게 한 것이 {score["sentence_sources"]["model_retry"]}개, 정해 둔 문장 틀로 바꾼 것이 {score["sentence_sources"]["template"]}개다.</li>
<li>그림만 본 모델은 식별기도, 갈라 주는 규칙도, 검사도 없이 같은 화면(그래픽을 가린 것)을 보고 중계한 것이다.</li>
</ul>

<h2>읽는 법</h2>
<ul>
<li><b>선두 중계</b>: 식별기가 번호를 말했고, 출발 뒤 8초가 지났고, 예상 결승까지 30초보다 많이 남은 시점. 선두권 번호를 넣어 말한다.</li>
<li><b>확정 정보</b>: 그 밖의 시점. 경주 정보, 지난 시간, 시계로 어림한 남은 거리만 말한다. 9초 안에 확인된 번호는 과거형으로만 말할 수 있다.</li>
<li>화면 그림의 상자는 식별기 결과다(파랑 = 앞선 말로 고름, 초록 = 번호 확정, 회색 = 번호 미확정). 그림에 보이는 순위 띠와 타이머는 보기 편하게 남긴 것이다.</li>
<li>말 이름, 선두권 안의 순서, 결승 순위는 시스템이 모른다. 그래서 문장이 실제 중계보다 밋밋하다.</li>
</ul>
<nav>{nav}</nav>
<div class="wrap">
{"".join(sections)}
</div>

<h2>한계</h2>
<ul>
<li>하루, {score["races"]}개 경주의 시연이다. 갈라 주는 규칙("예상 결승 30초 전까지")은 이번에 처음 쓴 것이고 따로 시험하지 않았다.</li>
<li>띠가 없는 시점(특히 마지막 30초)의 문장은 채점할 기준이 없다.</li>
<li>남은 거리는 시계로 어림한 값이다. 실제와 수십 미터 다를 수 있다.</li>
<li>식별에 시점 하나당 7~10초가 걸려 실시간이 아니다. 12번 이상의 말은 다루지 못한다.</li>
</ul>
</main></body></html>'''
(D / "commentary.html").write_text(page, encoding="utf-8")
print("wrote", D / "commentary.html", round(len(page) / 1e6, 2), "MB")
