"""Why do some frames get no correct leading-group statement? Development races only (exploration).

usage: python u1_b_diagnosis.py dev
Uses stripset/out/D0 and out_past/D0. No model calls.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "stripset"
split = sys.argv[1]
frames = [m for m in json.loads((S / "manifest.json").read_text())["frames"] if m["split"] == split]
READABLE = {1, 2, 3, 5, 6, 8, 9}   # colours Qwen names reliably; 4, 7, 10, 11+ it almost never names
TAGS = [None, "m05", "m10", "m15"]


def riders_of(frame, tag):
    p = (S / "out" / "D0" / f"{frame}.json") if tag is None else (S / "out_past" / "D0" / f"{frame}_{tag}.json")
    return json.loads(p.read_text())["riders"]


def votes(frame, W, K, states):
    v, ps = Counter(), {}
    for tag in TAGS[:W]:
        front = sorted(riders_of(frame, tag), key=lambda r: -r["box"][3])[:K]
        for r in front:
            if r["state"] in states and r["no"]:
                v[r["no"]] += 1
                ps.setdefault(r["no"], []).append(r["p_learned"] or 0.0)
    return v, ps


n = len(frames)
k = Counter()
top3_hard = Counter()
for m in frames:
    B3 = m["strip"][:3]
    sure_v, _ = votes(m["frame"], 3, 2, ("sure",))
    said = [x for x, c in sure_v.items() if c >= 2]
    ok = any(x in B3 for x in said)
    readable_in_top3 = sum(x in READABLE for x in B3)
    k["frames"] += 1
    k["correct_statement"] += ok
    k[f"readable_in_top3={readable_in_top3}"] += 1
    k[f"ok|readable_in_top3={readable_in_top3}"] += ok
    if not ok:
        k["missing"] += 1
        k["missing_and_leader_hard"] += m["strip"][0] not in READABLE
        k["missing_and_no_readable_top3"] += readable_in_top3 == 0
    for x in B3:
        top3_hard[x] += 1
print(f"split {split}: {n} frames | correct statement (3 frames, front 2, sure x2): {k['correct_statement']} ({k['correct_statement'] / n:.3f})")
for r in range(4):
    t = k[f"readable_in_top3={r}"]
    if t:
        print(f"  frames with {r} Qwen-readable numbers in the strip top 3: {t:3d} | with a correct statement: {k[f'ok|readable_in_top3={r}']}")
print(f"  frames with no correct statement: {k['missing']} | of these, strip #1 is a hard colour: {k['missing_and_leader_hard']} | no readable number in top 3: {k['missing_and_no_readable_top3']}")
print("  how often each number is in the strip top 3:", dict(sorted(top3_hard.items())))

# Could 'guess' answers that repeat over frames be trusted? Precision of repeated guesses for hard numbers.
print("\nrepeated GUESS answers among the front K boxes (any number), precision against strip top 3:")
print("W K m  min_p   statements  in top3   precision   frames gaining a correct statement (on top of sure x2 of 3, K2)")
for W in (3, 4):
    for K in (2, 3):
        for mth in (2, 3, 4):
            if mth > W:
                continue
            for minp in (0.0, 0.3, 0.5):
                st = ok = gain = 0
                for m in frames:
                    B3 = m["strip"][:3]
                    sure_v, _ = votes(m["frame"], 3, 2, ("sure",))
                    base = any(x in B3 for x, c in sure_v.items() if c >= 2)
                    gv, ps = votes(m["frame"], W, K, ("guess",))
                    said = [x for x, c in gv.items() if c >= mth and sum(ps[x]) / len(ps[x]) >= minp]
                    st += len(said)
                    ok += sum(x in B3 for x in said)
                    gain += (not base) and any(x in B3 for x in said)
                if st:
                    print(f"{W} {K} {mth}  {minp:.1f}     {st:4d}        {ok:4d}      {ok / st:.3f}       {gain}")
