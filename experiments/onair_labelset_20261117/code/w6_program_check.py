"""winner, after scoring: does the program's method give the calls of the calculation, when the finish is taken from the clock alone?
usage: w6_program_check.py OUT.json SCORE.json DAY [DAY ...]       (GPU; SCORE.json = what w4_winner_guard.py score wrote for these days)

For every race video: the finish as identify_l1.py --winner-call takes it (last change of the race clock minus 0.2 s; no official time),
the 31 pictures of the 6 s before it, LeadingGroupIdentifier.winner_call_checked with the v11 classifier. Compared with the calculation
(finish placed with the official winning time) and scored against the official winner."""
import bisect
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "identifier"))
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
from identify_l1 import TAU_LEADER_V9, WINNER_SPAN, LeadingGroupIdentifier, race_clock, video_pts

out, score, days = sys.argv[1], json.loads(Path(sys.argv[2]).read_text()), sys.argv[3:]
calc = {(r["day"], r["race"]): r for r in score["rows"]}
ident = LeadingGroupIdentifier(fast=True, fallback=True, tau_leader=TAU_LEADER_V9, cap_weights=ROOT / "cnn" / "models" / "caps2_a.pt")
rows, k = [], Counter()
for day in days:
    fc.FOLDER.setdefault(day, day)
    tms = fc.times(day)
    for race in sorted(tms):
        if (day, race) not in calc or "file" not in tms[race]:
            continue
        path = ROOT / day / "videos" / tms[race]["file"]
        S, F = race_clock(path)
        if F is None:
            rows.append({"day": day, "race": race, "program": None, "why": "no clock"})
            continue
        finish = round(F - 0.2, 3)
        pts = video_pts(path, ROOT / "winner" / "pts" / day / f"{race}.json")
        cap = cv2.VideoCapture(str(path))
        last = []
        for j in range(int(WINNER_SPAN * 5) + 1):
            i = bisect.bisect_right(pts, finish - WINNER_SPAN + 0.2 * j + 1e-8) - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if ok:
                last.append((bgr, round(WINNER_SPAN - 0.2 * j, 1)))
        cap.release()
        n, info = ident.winner_call_checked([b for b, _ in last], [t for _, t in last])
        c = calc[(day, race)]
        kind = "none" if n is None else ("right" if n == c["winner"] else "wrong")
        k[kind] += 1
        k["same as the calculation"] += n == c["new"]
        rows.append({"day": day, "race": race, "program": n, "calculation": c["new"], "winner": c["winner"], "clock_agrees": c["clock_agrees"], "finish_program": finish, **info})
        print(day, race, "program", n, "calculation", c["new"], "winner", c["winner"], info, flush=True)
print(len(rows), "races;", dict(k))
Path(out).write_text(json.dumps({"days": days, "totals": dict(k), "rows": rows}, indent=1))
