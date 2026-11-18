"""Finish stretch 2, stage 2: following riders from picture to picture, and the statement rule that uses it.

Idea: in the last seconds the cap of the leading rider is readable in some pictures and not in others. A rider is followed across
pictures taken 0.2 s apart (a "track"), the cap-colour probabilities seen along the track so far are averaged, and the number is
decided from that average. Everything is causal: a decision at time t uses pictures up to t only. The strip is never used.

Pieces
  link(pictures)                       -> track id for every box of every picture
  identify(pictures, ids, memory)      -> per picture: number and confidence of every box (memory=False: that picture alone)
  decide(...)                          -> per picture: numbers taken as leading, leader pick, scores
  statements(...)                      -> per time point: statements, leader (same voting / carry / fallback rules as identify_l1.py)"""
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

SURE, TAU, K = 0.7, 0.6, 2            # as in identify_l1.py
FALLBACK_SINGLE = 0.8


def link(pictures, gate=1.0, max_gap=3, min_matched=0.5):
    """Track ids. A box continues the track whose last box is nearest after removing the common shift of the picture (camera pan).
    gate: largest distance in rider heights. max_gap: pictures a track may be unseen. If fewer than `min_matched` of the boxes find
    a track, the picture is treated as a new shot and every box starts a new track."""
    tracks, nxt, out = {}, 0, []
    for k, pic in enumerate(pictures):
        B = np.array(pic["boxes"], float).reshape(-1, 4)
        ids = [-1] * len(B)
        c = np.c_[(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2] if len(B) else np.zeros((0, 2))
        h = B[:, 3] - B[:, 1] if len(B) else np.zeros(0)
        act = [i for i, t in tracks.items() if k - t["last"] <= max_gap]
        if len(B) and act:
            tc, th = np.array([tracks[i]["c"] for i in act]), np.array([tracks[i]["h"] for i in act])
            d = c[:, None, :] - tc[None, :, :]
            nn = np.argmin(np.linalg.norm(d, axis=2), axis=1)
            g = np.median(d[np.arange(len(B)), nn], axis=0) if len(B) >= 2 else np.zeros(2)
            size = np.abs(np.log(h[:, None] / th[None, :]))
            cost = np.linalg.norm(d - g, axis=2) / np.maximum(h[:, None], th[None, :]) + np.where(size > 0.5, 10.0, 0.0)
            ri, cj = linear_sum_assignment(cost)
            good = [(r, j) for r, j in zip(ri, cj) if cost[r, j] <= gate]
            if len(good) >= min_matched * min(len(B), len(act)):
                for r, j in good:
                    ids[r] = act[j]
        for r in range(len(B)):
            if ids[r] < 0:
                ids[r] = nxt
                nxt += 1
            tracks[ids[r]] = {"c": c[r], "h": h[r], "last": k}
        out.append(ids)
    return out


def identify(pictures, ids, memory=True, window=25):
    """Per picture: list of (number, confidence) per box. memory: average the probabilities of the last `window` sightings of the track."""
    seen, out = {}, []
    for pic, tid in zip(pictures, ids):
        P = np.array(pic["probs"], float).reshape(-1, 11)
        if not len(P):
            out.append([])
            continue
        M = P.copy()
        if memory:
            for r, i in enumerate(tid):
                seen.setdefault(i, []).append(P[r])
                M[r] = np.mean(seen[i][-window:], axis=0)
        ri, cj = linear_sum_assignment(-M)
        a = dict(zip(ri.tolist(), cj.tolist()))
        out.append([(a[r] + 1, float(M[r, a[r]])) if r in a else (0, 0.0) for r in range(len(P))])
    return out


def features(boxes, conf):
    """The 19 features of the fast configuration (identify_l1.py, nq_common.py)."""
    boxes = np.array(boxes, float)
    cx, yb, h = (boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3], boxes[:, 3] - boxes[:, 1]
    n = len(boxes)
    ry, rxr, rxl, rh = (np.argsort(np.argsort(v)) for v in (-yb, -cx, cx, -h))
    corr = float(np.corrcoef(cx, yb)[0, 1]) if n >= 3 and cx.std() > 1 and yb.std() > 1 else 0.0
    return [[ry[i], rxr[i], rxl[i], rh[i], ry[i] / n, rxr[i] / n, rh[i] / n, n, yb[i] / 1080, cx[i] / 1920, h[i] / 1080, h[i] / h.max(),
             (yb[i] - yb.mean()) / 1080, (cx[i] - cx.mean()) / 1920, cx.std() / 1920, yb.std() / 1080, corr, h.mean() / 1080, conf[i]] for i in range(n)]


def decide(pic, who, order_model, leader_model, sure=SURE, tau_leader=0.5):
    """One picture -> {"taken": numbers picked as leading (at most K), "lead3": number -> score, "first": number -> score, "pick": leader or None}."""
    res = {"taken": set(), "lead3": {}, "first": {}, "pick": None, "riders": len(pic["boxes"])}
    idx = [r for r, (num, p) in enumerate(who) if num and p >= sure]
    if not idx:
        return res
    F = np.array(features(pic["boxes"], [p for _, p in who]))[idx]
    pl = order_model.predict_proba(F)[:, 1]
    pf = leader_model.predict_proba(F)[:, 1]
    for j, r in enumerate(idx):
        res["lead3"][who[r][0]] = float(pl[j])
        res["first"][who[r][0]] = float(pf[j])
    for j in np.argsort(-pl)[:K]:
        if pl[j] >= TAU:
            res["taken"].add(who[idx[j]][0])
    j = int(np.argmax(pf))
    if pf[j] >= tau_leader:
        res["pick"] = who[idx[j]][0]
    return res


def statements(per_picture, taus, history, history_leader, offsets=(0.0, 0.6, 1.0), need=2, fallback=True):
    """per_picture: {to_finish (rounded to 0.1): decide() result}. taus: time points in decreasing order (seconds to the finish).
    history / history_leader: {to_finish: set / number} of earlier time points (from the 3 s program run), extended here.
    A number is stated when it was taken in at least `need` of the pictures at t, t+offsets (seconds earlier). Carry and fallback as in identify_l1.py."""
    out = []
    for tau in taus:
        pics = [per_picture[round(tau + o, 1)] for o in offsets if round(tau + o, 1) in per_picture]
        votes, lead_votes, top = Counter(), Counter(), {}
        for d in pics:
            for n in d["taken"]:
                votes[n] += 1
            if d["pick"] is not None:
                lead_votes[d["pick"]] += 1
            for n, v in d["lead3"].items():
                top[n] = max(top.get(n, 0.0), v)
        now = {n for n, v in votes.items() if v >= need}
        history[tau] = now
        prev = [history.get(tau + 3, set()), history.get(tau + 6, set())]
        carried = (prev[0] & prev[1]) - now
        added = set()
        if fallback and not (now | carried):
            added = prev[0] & set(top)
            if not added and top:
                n = max(top, key=top.get)
                if top[n] >= FALLBACK_SINGLE:
                    added = {n}
        picked = [n for n, v in lead_votes.most_common() if v >= need]
        lead = picked[0] if picked else None
        history_leader[tau] = lead
        source = "now" if lead is not None else None
        if lead is None:
            a, b = history_leader.get(tau + 3), history_leader.get(tau + 6)
            if a is not None and a == b:
                lead, source = a, "carried"
        detail = [{"number": n, "source": "now", "votes": votes[n]} for n in sorted(now)] + [{"number": n, "source": "carried"} for n in sorted(carried)] \
            + [{"number": n, "source": "fallback"} for n in sorted(added)]
        out.append({"to_finish": tau, "statements": sorted(now | carried | added), "detail": detail, "leader": lead, "leader_source": source,
                    "pictures": [{"riders": d["riders"], "detection": "", "sure": len(d["lead3"])} for d in pics]})
    return out
