"""Build review/REVIEW.html for the 178-rider label set (single self-contained file).

Reads labels/draft_labels.json, crops/ and sheets/*_boxes.jpg. Pure stdlib.
Pass --no-images to build a tiny copy for testing the page logic.
"""
import base64
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NO_IMAGES = "--no-images" in sys.argv
# --base batch2 builds the page for a later batch kept in a subfolder
BASE = ROOT / sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else ROOT
STORE_KEY = "onair_labelset_review_20261117" + ("" if BASE == ROOT else "_" + BASE.name)
BLANK = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=="
KO = {"white": "흰색", "yellow": "노랑", "red": "빨강", "black": "검정", "blue": "파랑", "green": "초록",
      "brown": "갈색", "pink": "분홍", "purple": "보라", "sky blue": "하늘색", "sky blue / white stripes": "하늘색 줄무늬"}
NOTE_KO = {
    "two riders in one box": "상자 하나에 기수 두 명",
    "two caps in one box (pink behind, red in front)": "상자 하나에 모자 두 개",
    "two caps in one box": "상자 하나에 모자 두 개",
    "possible duplicate box of p03 (same yellow cap, red silks)": "p03과 같은 기수일 수 있음",
    "possible duplicate box of p03": "p03과 같은 기수일 수 있음",
    "partial / possible duplicate of p03": "p03과 같은 기수일 수 있음",
    "blue cap with a stripe; may be 12 or higher": "줄무늬가 보임, 12번 이상일 수 있음",
    "second green cap; may be 12 or higher": "초록 모자가 둘, 12번 이상일 수 있음",
    "second red cap; may be 12 or higher": "빨강 모자가 둘, 12번 이상일 수 있음",
    "saddle cloth shows 3": "안장 번호판에 3이 보임",
}


def uri(rel):
    if NO_IMAGES:
        return BLANK
    return "data:image/jpeg;base64," + base64.b64encode((BASE / rel).read_bytes()).decode()


riders = json.loads((BASE / "labels" / "draft_labels.json").read_text(encoding="utf-8"))["riders"]
frames = defaultdict(list)
for r in riders:
    frames[r["frame"]].append(r)

data = {"frames": []}
for f in sorted(frames):
    items = sorted(frames[f], key=lambda r: r["id"])
    data["frames"].append({
        "id": f, "race": items[0]["race"], "pts": items[0]["pts"], "overview": uri(f"sheets/{f}_boxes.jpg"),
        "riders": [{"id": r["id"], "p": r["id"].rsplit("_", 1)[1], "num": r["draft_number"],
                    "colour": KO.get(r["draft_colour"], ""), "note": NOTE_KO.get(r["note"], r["note"]),
                    "img": uri(r["crop"])} for r in items],
    })

HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ON-AIR 기수 라벨 검수</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16202b;--mute:#5b6876;--line:#d7dde4;--ok:#1a7f4b;--okbg:#e3f5ea;
--fix:#b45309;--fixbg:#fdf0d9;--unk:#475569;--unkbg:#e8ecf1;--accent:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#12171d;--card:#1b222b;--ink:#e8edf2;--mute:#9aa7b4;--line:#2f3a46;
--ok:#4ade80;--okbg:#143424;--fix:#fbbf24;--fixbg:#3a2a0c;--unk:#cbd5e1;--unkbg:#2a333e;--accent:#7aa2ff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,"Apple SD Gothic Neo",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;
display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header h1{font-size:16px;margin:0;flex:1 1 200px}
.prog{font-variant-numeric:tabular-nums;color:var(--mute)}
main{max-width:1180px;margin:0 auto;padding:16px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:18px}
section h2{font-size:16px;margin:0 0 4px}
.hint{color:var(--mute);margin:0 0 10px;font-size:14px}
details{margin:0 0 10px}summary{cursor:pointer;color:var(--accent);font-size:14px}
details img{max-width:100%;height:auto;border-radius:6px;margin-top:8px;display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.card{border:1px solid var(--line);border-radius:8px;padding:8px;display:flex;flex-direction:column;gap:6px}
.card.ok{background:var(--okbg);border-color:var(--ok)}
.card.fix{background:var(--fixbg);border-color:var(--fix)}
.card.unknown{background:var(--unkbg);border-color:var(--unk)}
.card img{width:100%;height:170px;object-fit:contain;background:#1b2630;border-radius:6px}
.lab b{font-size:15px}.lab span{color:var(--mute);font-size:13px;display:block}
.btns{display:flex;gap:5px;flex-wrap:wrap}
button,select{font:inherit;font-size:14px;border:1px solid var(--line);background:var(--card);color:var(--ink);
border-radius:6px;padding:4px 8px;cursor:pointer}
button.on{font-weight:700;border-width:2px}
button.b-ok.on{border-color:var(--ok);color:var(--ok)}
button.b-unknown.on{border-color:var(--unk);color:var(--unk)}
select.on{border-color:var(--fix);color:var(--fix);border-width:2px;font-weight:700}
.all{margin:0 0 10px}
.primary{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:700}
textarea{width:100%;height:150px;font:12px/1.4 ui-monospace,Menlo,monospace;border:1px solid var(--line);
border-radius:8px;padding:8px;background:var(--bg);color:var(--ink)}
#msg{color:var(--ok);font-weight:700;margin-left:8px}
</style>
</head>
<body>
<header>
  <h1>ON-AIR 기수 라벨 검수</h1>
  <span class="prog" id="prog"></span>
  <button class="primary" onclick="document.getElementById('out').scrollIntoView({behavior:'smooth'})">결과로 이동</button>
</header>
<main>
<section>
  <h2>하는 법</h2>
  <p class="hint">사진마다 그 기수의 <b>모자 번호</b>가 맞는지 봐 주세요. 맞으면 <b>맞음</b>, 다르면 <b>올바른 번호</b>,
  알아볼 수 없으면 <b>미확인</b>입니다. 초안이 "판단 불가"인 사진은 번호를 아시면 골라 주시고, 모르시면 "판단 불가 맞음"을 누르세요.
  사진에 기수가 둘이면, 상자 가운데에 있는 기수 기준입니다. 애매하면 미확인으로 두세요.<br>
  모자색: 1 흰색 · 2 노랑 · 3 빨강 · 4 검정 · 5 파랑 · 6 초록 · 7 갈색 · 8 분홍 · 9 보라 · 10 하늘색 · 11 이상은 하늘색 바탕 줄무늬.<br>
  선택은 이 브라우저에 자동 저장됩니다. 끝나면 맨 아래 <b>결과 복사</b>를 눌러 대화창에 붙여 넣으세요. 일부만 해도 됩니다.</p>
</section>
<div id="app"></div>
<section id="out">
  <h2>결과</h2>
  <p class="hint">아래 내용을 복사해 대화창에 붙여 넣으세요.</p>
  <p><button class="primary" onclick="copyOut()">결과 복사</button><span id="msg"></span></p>
  <textarea id="ta" readonly></textarea>
</section>
</main>
<script>
const DATA = __DATA__;
const KEY = '__KEY__';
let S = {};
try { S = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { S = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch (e) {} };
const NUMS = Array.from({length: 16}, (_, i) => String(i + 1));
const OPTS = NUMS.map(n => `<option value="${n}">${n}번</option>`).join('');
const app = document.getElementById('app');
let total = 0;

for (const f of DATA.frames) {
  total += f.riders.length;
  const cards = f.riders.map(r => {
    const draft = r.num === null ? '초안: 판단 불가' : `초안: ${r.num}번 (${r.colour})`;
    return `<div class="card" data-id="${r.id}">
      <img src="${r.img}" alt="${r.id}">
      <div class="lab"><b>${r.p} · ${draft}</b>${r.note ? `<span>${r.note}</span>` : ''}</div>
      <div class="btns">
        <button class="b-ok" data-act="ok">${r.num === null ? '판단 불가 맞음' : '맞음'}</button>
        <select data-act="fix"><option value="">다른 번호…</option>${OPTS}</select>
        <button class="b-unknown" data-act="unknown">미확인</button>
      </div></div>`;
  }).join('');
  const s = document.createElement('section');
  s.innerHTML = `<h2>${f.race} · ${f.pts.toFixed(1)}초 · 기수 ${f.riders.length}명</h2>
    <details><summary>전체 화면에서 상자 위치 보기</summary><img src="${f.overview}" alt="${f.id} 전체 화면"></details>
    <p class="all"><button data-all="${f.riders.map(r => r.id).join(',')}">아직 안 고른 것 모두 초안대로</button></p>
    <div class="grid">${cards}</div>`;
  app.appendChild(s);
}

function paint() {
  document.querySelectorAll('.card').forEach(c => {
    const v = S[c.dataset.id];
    c.className = 'card' + (v ? ' ' + v.status : '');
    c.querySelector('[data-act=ok]').classList.toggle('on', !!v && v.status === 'ok');
    c.querySelector('[data-act=unknown]').classList.toggle('on', !!v && v.status === 'unknown');
    const sel = c.querySelector('select');
    sel.value = v && v.status === 'fix' ? v.number : '';
    sel.classList.toggle('on', !!v && v.status === 'fix');
  });
  const done = Object.keys(S).length;
  document.getElementById('prog').textContent = `검수 ${done} / ${total}`;
  document.getElementById('ta').value = 'ON-AIR 기수 라벨 검수 결과\n' + JSON.stringify({reviewed: done, total, riders: S});
}

app.addEventListener('click', e => {
  const b = e.target.closest('button');
  if (!b) return;
  if (b.dataset.all) {
    b.dataset.all.split(',').forEach(id => { if (!S[id]) S[id] = {status: 'ok'}; });
  } else if (b.dataset.act) {
    const id = b.closest('.card').dataset.id;
    if (S[id] && S[id].status === b.dataset.act) delete S[id]; else S[id] = {status: b.dataset.act};
  }
  save(); paint();
});
app.addEventListener('change', e => {
  if (e.target.tagName !== 'SELECT') return;
  const id = e.target.closest('.card').dataset.id;
  if (e.target.value) S[id] = {status: 'fix', number: e.target.value}; else delete S[id];
  save(); paint();
});

function copyOut() {
  const ta = document.getElementById('ta');
  const done = () => { document.getElementById('msg').textContent = '복사했습니다'; };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(ta.value).then(done, () => { ta.select(); document.execCommand('copy'); done(); });
  } else { ta.select(); document.execCommand('copy'); done(); }
}
paint();
</script>
</body>
</html>
"""

out = BASE / "review" / ("REVIEW_test.html" if NO_IMAGES else "REVIEW.html")
out.parent.mkdir(exist_ok=True)
out.write_text(HTML.replace("__KEY__", STORE_KEY).replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
print(out.name, out.stat().st_size, "bytes", "frames", len(data["frames"]), "riders", sum(len(f["riders"]) for f in data["frames"]))
