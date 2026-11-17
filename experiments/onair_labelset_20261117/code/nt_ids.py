"""Write DAY_DIR/video_ids.json for one Seoul race day from the cached playlist listing. usage: nt_ids.py DAY_DIR 2026-09-19 "RULE TEXT"
The listing (/tmp/onair_playlist.txt, lines "ID | (Seoul) 2026.09.19 R3") was made earlier from the KRBC International playlist."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
d, day, rule = sys.argv[1:4]
O = ROOT / d
O.mkdir(parents=True, exist_ok=True)
vids = {}
for line in Path("/tmp/onair_playlist.txt").read_text().splitlines():
    m = re.match(r"(\S+) \| \(Seoul\) " + re.escape(day.replace("-", ".")) + r" R(\d+)(\s.*)?$", line)
    if m:
        vids[int(m.group(2))] = {"race": f"R{int(m.group(2)):02d}", "id": m.group(1), "title": line.split(" | ", 1)[1].strip()}
assert vids and sorted(vids) == list(range(1, len(vids) + 1)), sorted(vids)
(O / "video_ids.json").write_text(json.dumps({"day": day, "place": "Seoul", "channel": "KRBC International", "chosen_by": rule, "videos": [vids[k] for k in sorted(vids)]}, indent=1, ensure_ascii=False))
print(d, day, "races", len(vids))
