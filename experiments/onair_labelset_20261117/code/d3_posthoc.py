"""POST-HOC look at the third-day result (after seeing otherday2/score.json). Not a pre-registered test.
Where do the wrong statements and the silent frames come from, on 2026-11-07 and 2026-11-14?"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent


def load(day_dir, probs):
    O = ROOT / day_dir
    man = json.loads((O / "manifest.json").read_text())
    return O, man["frames"], probs


def label(O, probs, name, past, thr=0.8, mode="C2"):
    riders = json.loads(((O / "out_past") if past else (O / "out")).joinpath(f"{name}.json").read_text())["riders"]
    if not riders:
        return []
    P = np.array(json.loads((O / probs / f"{name}.json").read_text())["probs"])
    Sc = P.copy()
    q = [r["qwen_number"] for r in riders]
    for i, n in enumerate(q):
        if n:
            Sc[i, n - 1] += 1.0
    ri, cj = linear_sum_assignment(-Sc)
    a = {int(i): int(j) for i, j in zip(ri, cj)}
    out = []
    for i, r in enumerate(riders):
        j = a.get(i)
        if j is None:
            out.append((r["box"][3], 0, False, r["box"]))
            continue
        agree = q[i] == j + 1
        conf = P[i, j] >= thr
        sure = (agree or conf) if mode == "C2" else (agree and conf) if mode == "AND" else conf
        out.append((r["box"][3], j + 1, bool(sure), r["box"]))
    return out


def run(day_dir, probs, thr=0.8, mode="C2", verbose=True):
    O, frames, probs = load(day_dir, probs)
    st = ok = ok5 = good = 0
    wrong45 = wrongoff = 0
    off = Counter()
    per = defaultdict(lambda: Counter())
    silent_small = silent = 0
    anysure_good = 0
    for m in frames:
        B = m["strip"]
        votes = Counter()
        allsure = set()
        nbox = 0
        for k, (name, past) in enumerate(((m["frame"], False), (f"{m['frame']}_m05", True), (f"{m['frame']}_m10", True))):
            L = label(O, probs, name, past, thr, mode)
            if k == 0:
                nbox = len(L)
                per[m["race"]]["boxes"] += len(L)
                per[m["race"]]["sure"] += sum(s for _, _, s, _ in L)
                per[m["race"]]["h"] += sum(b[3] - b[1] for _, _, _, b in L)
                allsure = {n for _, n, s, _ in L if s and n}
            for n in {n for _, n, s, _ in sorted(L, key=lambda x: -x[0])[:2] if s and n}:
                votes[n] += 1
        said = [n for n, v in votes.items() if v >= 2]
        st += len(said)
        for n in said:
            if n in B[:3]:
                ok += 1
            elif n in B:
                wrong45 += 1
            else:
                wrongoff += 1
                off[n] += 1
        ok5 += sum(n in B for n in said)
        g = any(n in B[:3] for n in said)
        good += g
        per[m["race"]]["frames"] += 1
        per[m["race"]]["said"] += len(said)
        per[m["race"]]["good"] += g
        if not g:
            silent += 1
            silent_small += nbox < 4
        anysure_good += any(n in B[:3] for n in allsure)
    A = ok / st if st else 0
    Bv = good / len(frames)
    if verbose:
        print(f"{day_dir} {probs} thr {thr} {mode}: frames {len(frames)} statements {st} A {A:.3f} B {Bv:.3f}")
        print(f"  wrong: in strip places 4-5: {wrong45}, not on strip: {wrongoff}; off-strip numbers said: {dict(sorted(off.items()))}")
        print(f"  frames without a right statement: {silent}, of which fewer than 4 rider boxes: {silent_small}")
        print(f"  ceiling if every sure number in the current frame were said: B {anysure_good/len(frames):.3f}")
        for r, c in sorted(per.items()):
            print(f"  {r}: frames {c['frames']:2d} boxes/frame {c['boxes']/c['frames']:.1f} mean box height {c['h']/max(1,c['boxes']):.0f}px sure share {c['sure']/max(1,c['boxes']):.2f} statements {c['said']:2d} good frames {c['good']:2d}")
    return st, A, Bv


if __name__ == "__main__":
    run("otherday2", "cnn_probs2")
    run("otherday2", "cnn_probs")
    run("otherday", "cnn_probs")
    print("--- threshold / rule sweep (post-hoc) ---")
    for d, p in (("otherday2", "cnn_probs2"), ("otherday2", "cnn_probs"), ("otherday", "cnn_probs")):
        for mode in ("C2", "CONF", "AND"):
            for thr in (0.8, 0.9, 0.95, 0.99):
                st, A, Bv = run(d, p, thr, mode, verbose=False)
                print(f"  {d:9s} {p:10s} {mode:4s} thr {thr:.2f}: statements {st:3d} A {A:.3f} B {Bv:.3f}")
