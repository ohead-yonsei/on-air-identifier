"""EXPLORATION on days already seen (not a test): carry statements over time within a race.
Base statement at a strip frame: sure among the two front boxes in >= 2 of the 3 most recent pictures (as now), with sure rule
C2 or CONF (classifier probability only). Temporal rules combine the base statements of frames t, t-3 s, t-6 s of the same race.
usage: tr1_temporal.py"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
DAYS = [("11-14", "otherday", "cnn_probs"), ("11-07", "otherday2", "cnn_probs_two"), ("10-24", "otherday3", "cnn_probs_two"), ("10-17", "otherday4", "cnn_probs_two")]
import sys
if len(sys.argv) > 1 and sys.argv[1] == "da":   # same days with adaptive detection (no DA run exists for 11-14)
    DAYS = [("11-07", "da/otherday2", "cnn_probs_two"), ("10-24", "da/otherday3", "cnn_probs_two"), ("10-17", "da/otherday4", "cnn_probs_two")]
if len(sys.argv) > 1 and sys.argv[1] == "d0three":
    DAYS = DAYS[1:]
CACHE = {}


def labels(O, folder, name, past):
    key = (str(O), name)
    if key not in CACHE:
        rs = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
        out = []
        if rs:
            P = np.array(json.loads((O / folder / f"{name}.json").read_text())["probs"])
            Sc = P.copy()
            q = [r["qwen_number"] for r in rs]
            for i, n in enumerate(q):
                if n:
                    Sc[i, n - 1] += 1.0
            ri, cj = linear_sum_assignment(-Sc)
            a = {int(i): int(j) for i, j in zip(ri, cj)}
            out = [(rs[i]["box"][3], a[i] + 1, float(P[i, a[i]]), q[i] == a[i] + 1) for i in range(len(rs)) if i in a]
        CACHE[key] = out
    return CACHE[key]


def base(O, folder, frame, mode, thr):
    votes = Counter()
    for name, past in ((frame, False), (f"{frame}_m05", True), (f"{frame}_m10", True)):
        for n in {n for _, n, p, ag in sorted(labels(O, folder, name, past), key=lambda x: -x[0])[:2] if ((ag or p >= thr) if mode == "C2" else p >= thr)}:
            votes[n] += 1
    return {n for n, v in votes.items() if v >= 2}


RULES = {
    "now": lambda s0, s1, s2: s0,
    "now|prev": lambda s0, s1, s2: s0 | s1,
    "now|prev|prev2": lambda s0, s1, s2: s0 | s1 | s2,
    "now&prev": lambda s0, s1, s2: s0 & s1,
    "now|(prev&prev2)": lambda s0, s1, s2: s0 | (s1 & s2),
    "2of3": lambda s0, s1, s2: {n for n in s0 | s1 | s2 if (n in s0) + (n in s1) + (n in s2) >= 2},
    "now|2of3": lambda s0, s1, s2: s0 | {n for n in s1 | s2 if (n in s1) + (n in s2) >= 2},
    "(now&prev)|(now&prev2)|(prev&prev2)": lambda s0, s1, s2: (s0 & s1) | (s0 & s2) | (s1 & s2),
}
table = {}
for mode, thr in (("C2", 0.8), ("CONF", 0.7), ("CONF", 0.8), ("CONF", 0.9)):
    for rname, rule in RULES.items():
        row = []
        for day, d, folder in DAYS:
            O = ROOT / d
            frames = json.loads((O / "manifest.json").read_text())["frames"]
            st = ok = good = 0
            for m in frames:
                race, t = m["frame"].split("_t")
                t = int(t)
                s = [base(O, folder, m["frame"], mode, thr)]
                for back in (3, 6):
                    prev = f"{race}_t{t - back:03d}"
                    s.append(base(O, folder, prev, mode, thr) if (O / "out" / f"{prev}.json").exists() else set())
                said = rule(*s)
                st += len(said)
                ok += sum(n in m["strip"][:3] for n in said)
                good += any(n in m["strip"][:3] for n in said)
            row.append((ok, st, good, len(frames)))
        table[(mode, thr, rname)] = row
print("sure rule | temporal rule | per day A / B (" + ", ".join(d[0] for d in DAYS) + ") | pooled A / B | worst-day A / B")
for (mode, thr, rname), row in sorted(table.items(), key=lambda kv: -(sum(r[0] for r in kv[1]) / max(1, sum(r[1] for r in kv[1])) + sum(r[2] for r in kv[1]) / sum(r[3] for r in kv[1]))):
    A = [r[0] / max(1, r[1]) for r in row]
    B = [r[2] / r[3] for r in row]
    pa, pb = sum(r[0] for r in row) / max(1, sum(r[1] for r in row)), sum(r[2] for r in row) / sum(r[3] for r in row)
    print(f"{mode} {thr} | {rname:36s} | " + "  ".join(f"{a:.2f}/{b:.2f}" for a, b in zip(A, B)) + f" | {pa:.3f}/{pb:.3f} | {min(A):.2f}/{min(B):.2f}")
