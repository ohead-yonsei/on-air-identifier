"""Build the team page from the template, with its four stills inlined as data URIs. usage: python3 build.py
Writes two files with the same content:
  onair_identifier.html         - fragment for the Claude artifact (the artifact host adds the document skeleton)
  <project root>/docs/index.html - complete HTML document for the GitHub Pages site"""
import base64
import re
from pathlib import Path

D = Path(__file__).resolve().parent
s = (D / "page.template.html").read_text(encoding="utf-8")
for name in ("overlay", "strip", "finish", "replay"):
    src = f'src="img/{name}.jpg"'
    assert src in s, name
    s = s.replace(src, 'src="data:image/jpeg;base64,' + base64.b64encode((D / "img" / f"{name}.jpg").read_bytes()).decode() + '"')
# table of contents: every <section> gets an id, and both lists (sidebar, folded list) are filled from the h2 headings
toc = []


def with_id(m):
    sid = m.group(1) or f"s{len(toc) + 1}"
    toc.append((sid, re.sub(r"<[^>]+>", "", m.group(3)).strip()))
    return f'<section id="{sid}">{m.group(2)}<h2>{m.group(3)}</h2>'


s = re.sub(r'<section(?: id="([^"]+)")?>(\s*)<h2>(.*?)</h2>', with_id, s)
items = "".join(f'<li><a href="#{i}">{t}</a></li>' for i, t in toc)
wide = ('<nav class="toc" aria-label="목차"><p class="toc-kicker">ON-AIR · 팀 공유</p><a class="toc-top" href="#top">선두권 말 번호 알아내기</a>'
        f'<ol>{items}</ol><p class="toc-more"><a href="https://ohead-yonsei.github.io/on-air-identifier/race-call-20261108-r11.html">중계 예시 페이지 →</a>'
        '<a href="https://github.com/ohead-yonsei/on-air-identifier">코드와 문서 (GitHub) →</a></p></nav>')
narrow = f'<details class="toc-m"><summary>목차 ({len(toc)}개 구간)</summary><ol>{items}</ol></details>'
for mark, html in (("<!-- toc-wide -->", wide), ("<!-- toc-narrow -->", narrow)):
    assert s.count(mark) == 1, mark
    s = s.replace(mark, html)
(D / "onair_identifier.html").write_text(s, encoding="utf-8")
print("wrote onair_identifier.html", round(len(s) / 1e6, 2), "MB")
head, body = s.split("<main>", 1)
site = D.parents[2] / "docs"
site.mkdir(exist_ok=True)
(site / "index.html").write_text('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                                 + head + "</head>\n<body>\n<main>" + body + "\n</body>\n</html>\n", encoding="utf-8")
(site / ".nojekyll").write_text("")
print("wrote", site / "index.html")
