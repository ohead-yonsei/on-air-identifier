"""POST-HOC exploration on the two days already seen (2026-11-14, 2026-11-07). Not a test.
Grid over: what counts as sure, how many front boxes, how many of the last 3 frames must agree.
Used to pick ONE rule to lock before a fourth day is downloaded."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
CACHE = {}


def riders(O, probs, name, past):
    key = (str(O), probs, name)
    if key not in CACHE:
        rs = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
        out = []
        if rs:
            P = np.array(json.loads((O / probs / f"{name}.json").read_text())["probs"])
            q = [r["qwen_number"] for r in rs]
            for use_q in (True, False):
                Sc = P.copy()
                if use_q:
                    for i, n in enumerate(q):
                        if n:
                            Sc[i, n - 1] += 1.0
                ri, cj = linear_sum_assignment(-Sc)
                a = {int(i): int(j) for i, j in zip(ri, cj)}
                out.append([(rs[i]["box"][3], a[i] + 1, float(P[i, a[i]]), q[i] == a[i] + 1) if i in a else (rs[i]["box"][3], 0, 0.0, False) for i in range(len(rs))])
        else:
            out = [[], []]
        CACHE[key] = out
    return CACHE[key]


def score(day_dir, probs, mode, thr, K, m_min):
    O = ROOT / day_dir
    frames = json.loads((O / "manifest.json").read_text())["frames"]
    st = ok = good = 0
    for m in frames:
        B = m["strip"]
        votes = Counter()
        for name, past in ((m["frame"], False), (f"{m['frame']}_m05", True), (f"{m['frame']}_m10", True)):
            L = riders(O, probs, name, past)[1 if mode == "CNN" else 0]
            L = sorted(L, key=lambda x: -x[0])[:K]
            for n in {n for _, n, p, ag in L if n and ((ag or p >= thr) if mode == "C2" else p >= thr)}:
                votes[n] += 1
        said = [n for n, v in votes.items() if v >= m_min]
        st += len(said)
        ok += sum(n in B[:3] for n in said)
        good += any(n in B[:3] for n in said)
    return st, (ok / st if st else 0.0), good / len(frames)


rows = []
for mode in ("C2", "CONF", "CNN"):
    for thr in (0.6, 0.7, 0.8, 0.9, 0.95):
        for K in (1, 2, 3, 4):
            for m_min in (1, 2, 3):
                a = score("otherday2", "cnn_probs2", mode, thr, K, m_min)
                b = score("otherday", "cnn_probs", mode, thr, K, m_min)
                c = score("otherday2", "cnn_probs", mode, thr, K, m_min)
                rows.append((mode, thr, K, m_min, a, b, c))
print("mode thr K m | 11-07 two-day: n A B | 11-14 one-day: n A B | 11-07 one-day: n A B")
for r in sorted(rows, key=lambda r: -(r[4][2] + r[5][2])):
    mode, thr, K, m_min, a, b, c = r
    if min(a[1], b[1]) >= 0.86:
        print(f"{mode:4s} {thr:.2f} {K} {m_min} | {a[0]:3d} {a[1]:.3f} {a[2]:.3f} | {b[0]:3d} {b[1]:.3f} {b[2]:.3f} | {c[0]:3d} {c[1]:.3f} {c[2]:.3f}")
print("--- best B with A >= 0.90 on both days ---")
ok = [r for r in rows if min(r[4][1], r[5][1]) >= 0.90]
for r in sorted(ok, key=lambda r: -(r[4][2] + r[5][2]))[:8]:
    print(r)
print("--- reference: current rule ---")
print([r for r in rows if r[:4] == ("C2", 0.8, 2, 2)])
