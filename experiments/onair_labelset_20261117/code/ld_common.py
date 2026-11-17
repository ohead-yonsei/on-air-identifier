"""Shared pieces for the learned "who is in front" (leader) rule. See leader/DESIGN_KO.md.
A leader statement at a time point: in each of the 3 most recent pictures one sure rider is picked as the leader (or none);
the number picked in at least 2 of the 3 is stated. With carry: if nothing is stated now and the same number was stated
at both t-3 s and t-6 s, that number is stated."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from or_common import THR  # noqa: E402  (sure = classifier probability >= 0.7)


def pick_hand(L):
    """Hand rule: the lowest box in the picture, if its number is sure."""
    for num, p, rank_y, _ in L:
        if rank_y == 0:
            return num if p >= THR else None
    return None


def pick_model(L, score_fn, tau):
    sure = [r for r in L if r[1] >= THR]
    if not sure:
        return None
    pr = score_fn(np.array([r[3] for r in sure]))
    i = int(np.argmax(pr))
    return sure[i][0] if pr[i] >= tau else None


def leader_statements(day_data, pick, carry):
    base = {}
    for m, pics in day_data:
        votes = Counter(n for n in (pick(L) for L in pics.values()) if n)
        best = [n for n, v in votes.items() if v >= 2]
        base[m["frame"]] = best[0] if best else None
    out = {}
    for m, _ in day_data:
        s = base[m["frame"]]
        if s is None and carry:
            race, t = m["frame"].split("_t")
            a, b = (base.get(f"{race}_t{int(t) - back:03d}") for back in (3, 6))
            if a is not None and a == b:
                s = a
        out[m["frame"]] = s
    return out


def tally(day_data, said):
    """points, statements, right (= strip's first), in the strip's first two, in the strip's first three"""
    n = st = ok = ok2 = ok3 = 0
    good = {}
    for m, _ in day_data:
        s = said[m["frame"]]
        n += 1
        good[m["frame"]] = bool(s is not None and s == m["strip"][0])
        if s is not None:
            st += 1
            ok += s == m["strip"][0]
            ok2 += s in m["strip"][:2]
            ok3 += s in m["strip"][:3]
    return {"points": n, "statements": st, "right": ok, "in_first_two": ok2, "in_first_three": ok3}, good


def number_scores(pics, fn):
    """Per number: mean of fn(feature matrix) over the pictures (of the 3) in which that number is sure."""
    acc = {}
    for L in pics.values():
        sure = [r for r in L if r[1] >= THR]
        if not sure:
            continue
        for r, v in zip(sure, fn(np.array([r[3] for r in sure]))):
            acc.setdefault(r[0], []).append(float(v))
    return {n: sum(v) / len(v) for n, v in acc.items()}


def yb_score(F):
    """Hand score: how low the box is in the picture (feature 8 = bottom edge / 1080). Lower box = ahead."""
    return F[:, 8]


def pair_statements(day_data, stated, fn, margin):
    """For every pair of numbers the identifier states at a time point (both scored in the current 3 pictures):
    say "a is ahead of b" when score(a) - score(b) >= margin. Returns {frame: [(ahead, behind)]} and the number of candidate pairs."""
    out, cand = {}, 0
    for m, pics in day_data:
        sc = number_scores(pics, fn)
        nums = sorted(n for n in stated[m["frame"]] if n in sc)
        said = []
        for i, a in enumerate(nums):
            for b in nums[i + 1:]:
                cand += 1
                if abs(sc[a] - sc[b]) >= margin:
                    said.append((a, b) if sc[a] > sc[b] else (b, a))
        out[m["frame"]] = said
    return out, cand


def pair_tally(day_data, pairs):
    """said pairs, scoreable (at least one of the two is on the strip), right (strip agrees; a number not on the strip counts as behind)"""
    said = scoreable = right = 0
    good = {}
    for m, _ in day_data:
        pos = {n: i for i, n in enumerate(m["strip"])}
        ok_here = []
        for a, b in pairs[m["frame"]]:
            said += 1
            if a not in pos and b not in pos:
                continue
            scoreable += 1
            ok = pos.get(a, 99) < pos.get(b, 99)
            right += ok
            ok_here.append(ok)
        good[m["frame"]] = ok_here
    return {"points": len(day_data), "said": said, "scoreable": scoreable, "right": right}, good
