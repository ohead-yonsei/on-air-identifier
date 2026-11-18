"""Finish stretch 2: shared pieces. Standard library only.

Reference without the running-order strip: the official passing times of every horse (600 m to go = G3F, fourth corner = C4,
200 m to go = G1F, finish). Between those points each horse is taken to move at constant speed, which gives every horse a distance
to go at any race time, hence an order. At G3F, C4, G1F and the finish the order is the official one (to the 0.1 s the sheet gives).
The race time of a video frame comes from the race clock on screen (fn2_times.py)."""
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
FOLDER = {**{f"otherday{s}": f"otherday{s}" for s in ("", 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)}, "20261115": "live_commentary/20261115"}
MARGIN_M = 3.0          # lenient reference: also right when within 3 m (a little over one length; 0.1 s is about 1.7 m) of the cut-off
WINDOWS = {"far": (12.0, 36.0), "last": (0.0, 12.0), "all": (0.0, 36.0)}     # seconds before the winner finishes: far = (12, 36], last = [0, 12]


def official(day):
    return json.loads((ROOT / "finish2" / "official" / f"{day}.json").read_text())["races"]


def times(day):
    return json.loads((ROOT / "finish2" / "times" / f"{day}.json").read_text())


def video_finish(tm, t_win):
    """Video time at which the winner finishes, from the clock. Returns (time or None, how, agree).
    The clock starts with the race in most videos and is stopped by hand, sometimes late; in a few it starts late and jumps to the
    official time at the finish (checked on pictures of three races, finish2/DESIGN_KO.md). agree = both ways give the same moment
    within 0.35 s; only those races are in the main analysis."""
    if not tm or "error" in tm or t_win is None:
        return None, "no clock", False
    run = tm["F"] - tm["S"]
    if abs(run - t_win - 0.15) <= 0.35:
        return round(tm["S"] + t_win - 0.1, 3), "clock start + official time", True     # the clock shows 0:00.1 at S
    if run > t_win:
        return round(tm["S"] + t_win - 0.1, 3), "clock start + official time (clock stopped late)", False
    return round(tm["F"] - 0.2, 3), "clock stop (clock started late)", False              # the last change of the digits comes about 0.2 s after the stop


def winner_time(race):
    w = [h for h in race["horses"] if h.get("rank") == 1 and h.get("time", {}).get("finish")]
    return w[0]["time"]["finish"] if w else None


def c4_to_go(race):
    fr = [(h["time"]["C4"] - h["time"]["G3F"]) / (h["time"]["G1F"] - h["time"]["G3F"]) for h in race["horses"]
          if h.get("time") and None not in (h["time"].get("G3F"), h["time"].get("C4"), h["time"].get("G1F")) and h["time"]["G1F"] > h["time"]["G3F"]]
    return round(600 - 400 * median(fr), 1) if fr else None


def knots(race):
    """number -> [(race time, metres to go), ...] for horses with G3F, G1F and finish times."""
    d4, out = c4_to_go(race), {}
    for h in race["horses"]:
        t = h.get("time") or {}
        if None in (t.get("G3F"), t.get("G1F"), t.get("finish")):
            continue
        k = [(t["G3F"], 600.0), (t["G1F"], 200.0), (t["finish"], 0.0)]
        if d4 is not None and t.get("C4") is not None and t["G3F"] < t["C4"] < t["G1F"] and 200 < d4 < 600:
            k.insert(1, (t["C4"], d4))
        out[h["number"]] = k
    return out


def to_go(k, r):
    for (t0, d0), (t1, d1) in zip(k, k[1:]):
        if r <= t1:
            break
    return d0 + (d1 - d0) * (r - t0) / (t1 - t0)        # extrapolates with the first / last speed outside the knots


def reference(race, r, margin=MARGIN_M):
    """Order at race time r. top3 / leader are strict; *_lenient add horses within `margin` metres of the third / of the first."""
    kn = knots(race)
    d = {n: to_go(k, r) for n, k in kn.items()}
    order = sorted(d, key=d.get)
    if len(order) < 3:
        return None
    return {"order": order, "top3": order[:3], "leader": order[0],
            "top3_lenient": [n for n in order if d[n] <= d[order[2]] + margin],
            "leader_lenient": [n for n in order if d[n] <= d[order[0]] + margin],
            "to_go": {n: round(v, 1) for n, v in d.items()}}


# ---- whole race (added 2026-11-18 after the stage-1 lock; the functions above are unchanged) ----

def knots_full(race):
    """number -> [(race time, metres to go), ...] from the start to the finish: start, S1F (200 m run), corners between S1F and G3F
    (distance estimated per race from the share of time the field needs), G3F, C4, G1F, finish."""
    D, late = race["distance"], knots(race)
    hs = [h for h in race["horses"] if h["number"] in late and (h.get("time") or {}).get("S1F")]
    est = {}
    for c in ("C1", "C2", "C3"):
        fr = [(h["time"][c] - h["time"]["S1F"]) / (h["time"]["G3F"] - h["time"]["S1F"]) for h in hs
              if h["time"].get(c) is not None and h["time"]["G3F"] > h["time"]["S1F"]]
        if fr and 0.03 < median(fr) < 0.97:
            est[c] = (D - 200) - ((D - 200) - 600) * median(fr)
    out = {}
    for h in hs:
        t = h["time"]
        k = [(0.0, float(D)), (t["S1F"], D - 200.0)]
        for c in ("C1", "C2", "C3"):
            if c in est and t.get(c) is not None and k[-1][0] < t[c] < t["G3F"] and 600 < est[c] < k[-1][1]:
                k.append((t[c], est[c]))
        k += [x for x in late[h["number"]] if x[0] > k[-1][0]]
        out[h["number"]] = k
    return out


def reference_full(race, r, margin=MARGIN_M):
    """As reference(), for any race time from the start on."""
    d = {n: to_go(k, r) for n, k in knots_full(race).items()}
    order = sorted(d, key=d.get)
    if len(order) < 3:
        return None
    return {"order": order, "top3": order[:3], "leader": order[0],
            "top3_lenient": [n for n in order if d[n] <= d[order[2]] + margin],
            "leader_lenient": [n for n in order if d[n] <= d[order[0]] + margin],
            "to_go": {n: round(v, 1) for n, v in d.items()}}
