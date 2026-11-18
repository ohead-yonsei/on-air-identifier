"""Finish stretch 2, step 1: official results (finishing order, passing order and passing times of every horse).
usage: fn1_official.py OUT_DIR RAW_DIR

Source: the public result sheet of the Korea Racing Authority, one page per race
(race.kra.co.kr/raceScore/ScoretableDetailList.do?meet=1&realRcDate=YYYYMMDD&realRcNo=N; meet 1 = Seoul).
Only numbers are kept (horse number, rank, passing ranks, times). Names of horses, riders, trainers and owners are not stored.
The raw pages go to RAW_DIR (outside the project) and are not kept in the repository. Standard library only."""
import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

DAYS = {"otherday": "2026-11-14", "otherday2": "2026-11-07", "otherday3": "2026-10-24", "otherday4": "2026-10-17",
        "otherday5": "2026-10-10", "otherday6": "2026-10-05", "otherday7": "2026-10-03", "otherday8": "2026-09-19",
        "otherday9": "2026-09-12", "otherday10": "2026-09-05", "otherday11": "2026-08-30", "20261115": "2026-11-15"}
RACES = {"20261115": 11}
URL = "https://race.kra.co.kr/raceScore/ScoretableDetailList.do?meet=1&realRcDate={d}&realRcNo={n}"
POINTS = ["S1F", "C1", "C2", "C3", "G3F", "C4", "G1F"]


def secs(s):
    m = re.fullmatch(r"(\d+):(\d+\.\d)", s.strip())
    return round(int(m.group(1)) * 60 + float(m.group(2)), 1) if m else None


def cells(row):
    return [re.sub(r"\s+", " ", html.unescape(re.sub(r"<.*?>", "", c))).strip() for c in re.findall(r"<t[hd].*?</t[hd]>", row, flags=re.S)]


def tables(text):
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", text, flags=re.S)
    return [[cells(r) for r in re.findall(r"<tr.*?</tr>", tb, flags=re.S)] for tb in re.findall(r"<table.*?</table>", text, flags=re.S)]


def parse(text):
    tb = tables(text)
    head = " ".join(" ".join(r) for r in tb[0])
    dist = int(re.search(r"(\d{3,4})M", head).group(1))
    result = next(t for t in tb if t and t[0][:2] == ["순위", "마번"] and "마명" in t[0])
    passing = next(t for t in tb if t and t[0][:2] == ["순위", "마번"] and any("통과" in c for c in t[0]))
    horses = {}
    for r in result[1:]:
        if len(r) < 2 or not r[1].isdigit():
            continue
        horses[int(r[1])] = {"number": int(r[1]), "rank": int(r[0]) if r[0].isdigit() else None, "rank_text": r[0]}
    for r in passing[2:]:
        if len(r) < 13 or not r[1].isdigit():
            continue
        h = horses.setdefault(int(r[1]), {"number": int(r[1]), "rank": int(r[0]) if r[0].isdigit() else None, "rank_text": r[0]})
        ranks = [x.strip() for x in r[2].split("-")]
        h["pass_rank"] = {p: (int(x) if x.isdigit() else None) for p, x in zip(POINTS, ranks)}
        h["time"] = {"S1F": secs(r[3]), "C1": secs(r[4]), "C2": secs(r[5]), "C3": secs(r[6]), "G3F": secs(r[7]), "C4": secs(r[8]),
                     "G1F": secs(r[9]), "finish": secs(r[12])}
        h["last_3f"], h["last_1f"] = secs(r[10]), secs(r[11])
    lead = {}
    for t in tb:
        rows = {r[0]: r[1:] for r in t if r}
        if "통과거리" in rows and "통과타임" in rows:
            lead = {int(d): secs(x) for d, x in zip(rows["통과거리"], rows["통과타임"]) if d.isdigit()}
    return {"distance": dist, "horses": sorted(horses.values(), key=lambda h: (h["rank"] is None, h["rank"] or 0, h["number"])),
            "leader_time_at": lead}


def main():
    out, raw = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    for day, date in DAYS.items():
        res = {"day": day, "date": date, "place": "Seoul", "source": "Korea Racing Authority public result sheets (race.kra.co.kr), numbers only",
               "races": {}}
        for n in range(1, RACES.get(day, 10) + 1):
            f = raw / f"{day}_R{n:02d}.html"
            if not f.exists() or f.stat().st_size < 20000:
                subprocess.run(["curl", "-s", "-A", "Mozilla/5.0 (research script)", "-o", str(f), URL.format(d=date.replace("-", ""), n=n)], check=True)
                time.sleep(1.0)
            try:
                res["races"][f"R{n:02d}"] = parse(f.read_bytes().decode("euc-kr", errors="replace"))
            except Exception as e:      # noqa: BLE001
                res["races"][f"R{n:02d}"] = {"error": repr(e)}
        (out / f"{day}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        ok = [r for r, v in res["races"].items() if "error" not in v]
        print(day, date, len(ok), "races;", " ".join(f"{r}:{v['distance']}m/{len(v['horses'])}" for r, v in res["races"].items() if "error" not in v),
              [r for r in res["races"] if r not in ok], flush=True)


if __name__ == "__main__":
    main()
