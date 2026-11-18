"""Script demo, step 1: race card (what a commentator knows before the race) and official result (for scoring afterwards) of one race.
usage: ls1_racecard.py DATE(YYYYMMDD) RACE_NO OUT_DIR RAW_DIR

Source: the public result sheet of the Korea Racing Authority (same page as fn1_official.py).
racecard.json  number -> horse name, distance, runners. This is all the commentary system may read. Horse names only
               (no rider, trainer or owner names are stored).
official.json  ranks and passing times, numbers only (fn1_official.parse). Never read by the commentary system."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn1_official as f1

date, n, out, raw = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
raw.mkdir(parents=True, exist_ok=True)
f = raw / f"{date}_R{n:02d}.html"
if not f.exists() or f.stat().st_size < 20000:
    subprocess.run(["curl", "-s", "-A", "Mozilla/5.0 (research script)", "-o", str(f), f1.URL.format(d=date, n=n)], check=True)
text = f.read_bytes().decode("euc-kr", errors="replace")
off = f1.parse(text)
result = next(t for t in f1.tables(text) if t and t[0][:2] == ["순위", "마번"] and "마명" in t[0])
names = {int(r[1]): r[2] for r in result[1:] if len(r) > 2 and r[1].isdigit()}
ran = sorted(h["number"] for h in off["horses"] if (h.get("time") or {}).get("S1F") is not None or h.get("rank") is not None)
out.mkdir(parents=True, exist_ok=True)
(out / "racecard.json").write_text(json.dumps({"date": f"{date[:4]}-{date[4:6]}-{date[6:]}", "place": "서울", "race": n, "distance": off["distance"],
                                              "runners": [{"number": k, "name": names[k]} for k in ran],
                                              "note": "number, horse name and distance as on the race card; taken from the public result sheet"}, ensure_ascii=False, indent=1))
(out / "official.json").write_text(json.dumps({"date": date, "race": n, **off}, ensure_ascii=False, indent=1))
print(off["distance"], "m,", len(ran), "runners:", ", ".join(f"{k} {names[k]}" for k in ran))
