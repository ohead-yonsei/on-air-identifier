"""Build the team page from the template, with its four stills inlined as data URIs. usage: python3 build.py
Writes two files with the same content:
  onair_identifier.html         - fragment for the Claude artifact (the artifact host adds the document skeleton)
  <project root>/docs/index.html - complete HTML document for the GitHub Pages site"""
import base64
from pathlib import Path

D = Path(__file__).resolve().parent
s = (D / "page.template.html").read_text(encoding="utf-8")
for name in ("overlay", "strip", "finish", "replay"):
    src = f'src="img/{name}.jpg"'
    assert src in s, name
    s = s.replace(src, 'src="data:image/jpeg;base64,' + base64.b64encode((D / "img" / f"{name}.jpg").read_bytes()).decode() + '"')
(D / "onair_identifier.html").write_text(s, encoding="utf-8")
print("wrote onair_identifier.html", round(len(s) / 1e6, 2), "MB")
head, body = s.split("<main>", 1)
site = D.parents[2] / "docs"
site.mkdir(exist_ok=True)
(site / "index.html").write_text('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                                 + head + "</head>\n<body>\n<main>" + body + "\n</body>\n</html>\n", encoding="utf-8")
(site / ".nojekyll").write_text("")
print("wrote", site / "index.html")
