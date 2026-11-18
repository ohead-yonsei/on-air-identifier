"""Script demo, step 5: one page with the pictures and the script side by side.
usage: python3 code/ls5_page.py DEMO_DIR        (standard library only)

Reads script.json, score.json, ident.json, racecard.json and frames/*.jpg; writes DEMO_DIR/commentary.html
(a page fragment for the Claude artifact host: title and style first, no html/head/body tags; pictures inlined)."""
import base64
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / sys.argv[1]
script, score, ident, card = (json.loads((D / f"{n}.json").read_text()) for n in ("script", "score", "ident", "racecard"))
NAME = {r["number"]: r["name"] for r in card["runners"]}
CAP = {1: "흰색", 2: "노란색", 3: "빨간색", 4: "검정", 5: "파란색", 6: "초록색", 7: "갈색", 8: "분홍색", 9: "보라색", 10: "하늘색"}
WHY = {"empty, too long or a banned word": "비었거나 너무 길거나, 알 수 없는 것(순위·기수·주로 위치 등)을 말함", "a number that is not allowed": "식별기가 주지 않은 번호",
       "a name without its number, or a horse that is not allowed": "번호 없이 이름만 부르거나, 주지 않은 말을 부름", "no number in a ranking sentence": "순위 문장인데 말을 부르지 않음",
       "a leader word without a given leader": "선두를 주지 않았는데 선두를 말함", "leader claimed for another horse": "다른 말을 선두라고 함",
       "following words without a known order": "앞뒤 순서를 모르는데 '뒤따른다'고 함", "winner without a winner call": "우승마를 부르지 않았는데 우승을 말함",
       "the finish or a win before the clock stopped": "결승 전에 결승선 통과나 승리를 말함", "passing words without a change of leader": "선두가 바뀌지 않았는데 추월을 말함",
       "a word about gaps or a duel": "거리 차이나 접전을 말함 (알 수 없는 것)", "a time or a phase name read out": "주지 않은 시간이나 구간 이름을 말함",
       "a distance that was not given": "주지 않은 거리를 말함", "the same sentence as just before": "바로 앞과 같은 문장"}
e = html.escape


def chip(n):
    return f'<span class="cap c{n if n <= 10 else 11}" title="{CAP.get(n, "하늘색 줄무늬")} 모자">{n}</span>'


def horse(n):
    return f'{chip(n)}<span class="hn">{e(NAME.get(n, ""))}</span>'


def clock(sec):
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


sc = {r["k"]: r for r in score["rows"]}
tot = score["totals"]
beats = []
for r in script["rows"]:
    s = sc[r["k"]]
    img = base64.b64encode((D / "frames" / r["frame"]).read_bytes()).decode()
    g = r["given"]
    mode = {"ranking": "순위 중계", "confirmed": "확정 정보", "finish": "결승"}[r["mode"]]
    ev = []
    if r["mode"] == "finish":
        ev.append("식별기가 준 것: " + (f"우승마 {horse(g['winner'])}" if g["winner"] else "우승마를 부르지 못함"))
    elif g["leader"] is not None or g["group"]:
        parts = ([f"선두 {horse(g['leader'])}"] if g["leader"] is not None else []) + ([("뒤따르는 말 " if g.get("follow_ok") else "선두권 ") + " ".join(horse(n) for n in g["group"])] if g["group"] else [])
        ev.append("식별기가 준 것: " + " · ".join(parts))
    else:
        ev.append("식별기가 준 것: 없음" + (" (출발 직후에는 쓰지 않음)" if r["elapsed"] < 9 else ""))
    verdict = ""
    if "winner_call" in s:
        w = s["winner_call"]
        verdict = (f'<span class="v ok">맞음</span> 공식 우승마 {horse(w["official_winner"])}' if w["verdict"] == "right" else
                   f'<span class="v bad">틀림</span> 공식 우승마 {horse(w["official_winner"])}' if w["verdict"] == "wrong" else f'<span class="v na">부르지 않음</span> 공식 우승마 {horse(w["official_winner"])}')
    elif s["top3"]:
        marks = []
        for n, v in s["numbers"].items():
            marks.append(f'{chip(int(n))}<span class="v {"ok" if v == "right" else "near" if v == "close" else "bad"}">{"맞음" if v == "right" else "3m 안" if v == "close" else "틀림"}</span>')
        if s["leader"]:
            v = s["leader"]["verdict"]
            marks.append(f'선두 주장 <span class="v {"ok" if v == "right" else "near" if v == "close" else "bad"}">{"맞음" if v == "right" else "선두와 3m 안" if v == "close" else "틀림"}</span>')
        verdict = "공식 기록으로 본 그 순간 1~3위 " + " ".join(chip(n) for n in s["top3"]) + ((" → " + " ".join(marks)) if marks else "")
    rej = [t for t in r["tries"] if t["rejected"]]
    gate = ""
    if rej:
        gate = f'<details><summary>코드가 걸러낸 문장 {len(rej)}개' + (" → 정해진 틀로 대신함" if r["source"] == "template" else "") + '</summary><ul>' + "".join(
            f'<li><span class="rl">{e(t["line"]) or "(빈 답)"}</span><span class="rw">{e(WHY.get(t["rejected"], "검수 모델: " + t["rejected"].split(": ", 1)[-1] if t["rejected"].startswith("the checking model") else t["rejected"]))}</span></li>' for t in rej) + "</ul></details>"
    beats.append(f'''<article class="beat m-{r["mode"]}">
  <div class="tc"><span class="clk">{clock(r["elapsed"])}</span><span class="md">{mode}</span></div>
  <img src="data:image/jpeg;base64,{img}" width="640" height="360" alt="출발 {int(r["elapsed"])}초 뒤의 중계 화면" loading="lazy">
  <div class="tx"><p class="line">{e(r["line"])}</p><p class="ev">{ev[0]}</p>{f'<p class="ev">{verdict}</p>' if verdict else ""}{gate}</div>
</article>''')

n_model = sum(r["source"] != "template" for r in script["rows"])
w = score["winner_call"]
page = f'''<title>서울 11경주 AI 중계</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=Noto+Sans+KR:wght@400;500;700&family=Noto+Serif+KR:wght@600;700&display=swap">
<style>
/* layout: a timecoded call sheet. left rail = race clock, middle = the picture, right = the line as spoken with its evidence underneath */
:root {{ --bg:#f5f7fa; --panel:#ffffff; --ink:#101b2d; --quiet:#5a6a82; --line:#d9e1ec; --accent:#1553b8; --tint:#e8eef8; --ok:#17794a; --near:#8a6100; --bad:#b3261e;
  --sans:"Noto Sans KR",system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif; --serif:"Noto Serif KR","Apple SD Gothic Neo",serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0c131f; --panel:#131d2c; --ink:#e7edf6; --quiet:#94a4bb; --line:#26344b; --accent:#74a9ff; --tint:#172438; --ok:#55c891; --near:#e2b955; --bad:#ff8f85; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#0c131f; --panel:#131d2c; --ink:#e7edf6; --quiet:#94a4bb; --line:#26344b; --accent:#74a9ff; --tint:#172438; --ok:#55c891; --near:#e2b955; --bad:#ff8f85; color-scheme:dark }}
body {{ background:var(--bg); color:var(--ink); font-family:var(--sans); font-size:16px; line-height:1.7; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); padding-block:36px 64px; }}
main {{ max-width:980px; margin:0 auto; display:flex; flex-direction:column; gap:36px; }}
h1 {{ font-family:var(--serif); font-size:clamp(26px,4.4vw,38px); line-height:1.3; margin:6px 0 12px; text-wrap:balance; }}
h2 {{ font-size:19px; margin:0 0 10px; }}
p {{ margin:0; }} .lead {{ max-width:42em; color:var(--ink); }} .q {{ color:var(--quiet); }}
.eyebrow {{ font-family:var(--mono); font-size:12.5px; letter-spacing:.06em; color:var(--accent); }}
.facts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
.fact {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px 16px; }}
.fact b {{ display:block; font-family:var(--mono); font-size:24px; line-height:1.2; font-variant-numeric:tabular-nums; }} .fact span {{ font-size:13.5px; color:var(--quiet); }}
.card {{ display:flex; flex-wrap:wrap; gap:8px 18px; }} .card span.r {{ white-space:nowrap; }}
.cap {{ display:inline-flex; align-items:center; justify-content:center; min-width:1.55em; height:1.55em; padding:0 .25em; border-radius:999px; font-family:var(--mono); font-size:.8em; font-weight:600; border:1px solid var(--line); margin-right:.3em; vertical-align:.05em; }}
.c1 {{ background:#fff; color:#111 }} .c2 {{ background:#f2cf2b; color:#111 }} .c3 {{ background:#d6342c; color:#fff }} .c4 {{ background:#16181c; color:#fff }} .c5 {{ background:#1e55c8; color:#fff }}
.c6 {{ background:#1f9a55; color:#fff }} .c7 {{ background:#7a4a24; color:#fff }} .c8 {{ background:#f08fb6; color:#111 }} .c9 {{ background:#7a3fb5; color:#fff }} .c10 {{ background:#7cc7f2; color:#111 }} .c11 {{ background:repeating-linear-gradient(45deg,#7cc7f2 0 4px,#fff 4px 7px); color:#111 }}
.hn {{ margin-right:.5em; }}
ol.how {{ margin:0; padding-left:1.3em; max-width:44em; }} ol.how li {{ margin:4px 0; }}
.sheet {{ display:flex; flex-direction:column; }}
.beat {{ display:grid; grid-template-columns:76px minmax(0,300px) minmax(0,1fr); gap:18px; padding:18px 0; border-top:1px solid var(--line); align-items:start; }}
.beat img {{ width:100%; height:auto; border-radius:4px; display:block; background:var(--tint); }}
.tc {{ display:flex; flex-direction:column; gap:4px; }} .clk {{ font-family:var(--mono); font-size:17px; font-weight:600; font-variant-numeric:tabular-nums; }}
.md {{ font-size:12px; color:var(--quiet); }} .m-ranking .md {{ color:var(--accent); font-weight:700; }} .m-finish .md {{ color:var(--ok); font-weight:700; }}
.tx {{ min-width:0; display:flex; flex-direction:column; gap:8px; }}
.line {{ font-family:var(--serif); font-weight:600; font-size:clamp(19px,2.4vw,23px); line-height:1.5; text-wrap:pretty; }}
.m-confirmed .line {{ color:var(--quiet); }} .m-finish .line {{ font-size:clamp(22px,3vw,28px); }}
.ev {{ font-size:13.5px; color:var(--quiet); line-height:1.9; }}
.v {{ font-weight:700; font-size:12.5px; margin:0 .6em 0 .15em; }} .v.ok {{ color:var(--ok) }} .v.near {{ color:var(--near) }} .v.bad {{ color:var(--bad) }} .v.na {{ color:var(--quiet) }}
details {{ font-size:13.5px; color:var(--quiet); }} summary {{ cursor:pointer; width:fit-content; }} summary:focus-visible {{ outline:2px solid var(--accent); outline-offset:3px; }}
details ul {{ margin:6px 0 0; padding-left:1.2em; display:flex; flex-direction:column; gap:4px; }} .rl {{ text-decoration:line-through; text-decoration-color:var(--bad); margin-right:.6em; }} .rw {{ color:var(--bad); }}
.note {{ border-left:3px solid var(--accent); padding:2px 0 2px 16px; max-width:44em; }} ul.lim {{ margin:0; padding-left:1.2em; max-width:44em; }} ul.lim li {{ margin:5px 0; }}
@media (max-width:720px) {{ .beat {{ grid-template-columns:1fr; gap:10px; }} .tc {{ flex-direction:row; align-items:baseline; gap:12px; }} }}
</style>
<main>
<header>
  <div class="eyebrow">ON-AIR 시연 · {card["date"]} · 서울 {card["race"]}경주 {card["distance"]}m</div>
  <h1>띠 없이 화면만 보고 쓴 중계 스크립트</h1>
  <p class="lead">한 번도 쓰지 않은 경주 영상 하나를 받아, 식별기가 화면에서 알아낸 것만으로 로컬 언어 모델이 3초마다 한 문장씩 중계를 썼습니다. 순위 띠, 공식 결과, 방송의 소리는 쓰지 않았습니다. 아래에서 그때의 화면과 문장을 나란히 읽을 수 있습니다.</p>
</header>
<section class="facts">
  <div class="fact"><b>{tot["lines"]}문장</b><span>출발부터 결승까지. 그중 {tot["ranking_lines"]}문장이 말을 부른 순위 중계</span></div>
  <div class="fact"><b>{tot["numbers_right"]} / {tot["numbers"]}</b><span>부른 말이 그 순간 실제 1~3위였던 수{f" (나머지 {tot['numbers_close']}개는 3위와 3m 안)" if tot["numbers_close"] and not tot["numbers_wrong"] else ""}</span></div>
  <div class="fact"><b>{tot["leader_right"]} / {tot["leader_claims"]}</b><span>"선두"라고 한 말이 그 순간 실제 선두였던 수</span></div>
  <div class="fact"><b>{"우승마 맞음" if w["verdict"] == "right" else "우승마 틀림" if w["verdict"] == "wrong" else "우승마 못 부름"}</b><span>결승 순간에 부른 말{": " + str(w["number"]) + "번 " + e(NAME[w["number"]]) if w["number"] else " 없음"}</span></div>
</section>
<section>
  <h2>출전표</h2>
  <p class="card">{"".join(f'<span class="r">{horse(r["number"])}</span>' for r in card["runners"])}</p>
  <p class="q" style="font-size:13.5px;margin-top:8px">번호 옆의 색은 그 번호 기수의 모자색입니다. 식별기는 이 색으로 번호를 알아냅니다. 말 이름은 출전표에서 온 확정 정보입니다.</p>
</section>
<section>
  <h2>어떻게 만들었나</h2>
  <ol class="how">
    <li><b>화면 → 사실.</b> 식별기 v9가 3초마다 그림 3장을 보고 선두권 번호와 선두를 냅니다. 결승 36초쯤 전부터는 공식 기록으로 배운 선두 모델을 쓰고, 결승 순간에는 마지막 6초의 그림으로 우승마를 부릅니다.</li>
    <li><b>가르기.</b> 식별기가 확실하다고 한 것만 "말해도 되는 사실"로 넘깁니다. 없으면 경주 정보와 남은 거리만 넘깁니다.</li>
    <li><b>문장 쓰기.</b> 로컬 언어 모델({e(script["model"].split(":")[0])})이 그 사실만 보고 캐스터 말투로 씁니다. 그림은 보지 않습니다.</li>
    <li><b>검사.</b> 코드가 문장을 검사해, 주지 않은 말·선두·거리 차이·추월 등을 말했으면 버리고 다시 시킵니다. 네 번 안에 통과하지 못하면 정해진 틀의 문장을 씁니다.</li>
  </ol>
  <p class="q" style="margin-top:10px;max-width:44em">문장마다 아래에 "식별기가 준 것"과, 나중에 공식 기록으로 확인한 그 순간의 실제 1~3위를 적었습니다. 그림 속 초록 상자는 식별기가 그 순간 선두권으로 말한 기수입니다. 그림 아래쪽의 순위 띠는 방송 화면에 원래 있는 것이고, 시스템은 그 부분을 가리고 봅니다.</p>
</section>
<section>
  <h2>중계 스크립트</h2>
  <div class="sheet">
{chr(10).join(beats)}
  </div>
</section>
<section>
  <h2>읽고 나서 알아 둘 것</h2>
  <ul class="lim">
    <li><b>경주 하나입니다.</b> 이 경주에서 잘 맞았다고 늘 이렇다는 뜻은 아닙니다. 여러 날로 잰 값은 번호 정확도 93%, 선두 문장 88~90%입니다.</li>
    <li><b>{tot["lines"]}문장 중 {n_model}문장을 모델이 썼고, {tot["lines"] - n_model}문장은 정해진 틀입니다.</b> 모델이 "바짝 추격", "접전" 같은 말을 자꾸 쓰려 했고, 거리 차이는 식별기가 알려 주지 않는 것이라 코드가 걸러 냈습니다. 그래서 실제 중계보다 밋밋합니다.</li>
    <li><b>말하지 못하는 것이 많습니다.</b> 말 사이의 거리, 코너와 직선주로, 뒤쪽 말들의 움직임, 기수는 다루지 않습니다. 선두권 세 마리쯤만 말합니다.</li>
    <li><b>남은 거리는 어림값입니다.</b> 경주 거리와 예상 주파 시간으로 계산했습니다.</li>
    <li><b>실시간으로 돌린 것은 아닙니다.</b> 녹화 영상을 앞에서부터 차례로 처리했고, 문장 하나를 쓰는 데 {script["seconds_per_call_median"]}초쯤 걸렸습니다. 소리로 읽어 주는 단계는 없습니다.</li>
    <li><b>영상의 원래 중계 소리는 받아만 두었습니다.</b> 시스템의 입력으로 쓰지 않았고, 원래 중계와 비교하는 일도 아직 하지 않았습니다.</li>
  </ul>
</section>
</main>
'''
(D / "commentary.html").write_text(page, encoding="utf-8")
print("wrote", D / "commentary.html", round(len(page) / 1e6, 2), "MB;", tot)
