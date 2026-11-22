"""ON-AIR live run: read a race video at its own speed and put out one line of commentary every few seconds, as a live system would.
usage (on the server): python live_run.py --video RACE.mp4 --racecard racecard.json --out-dir OUT [--no-wait] [--no-names] [--step 3]

What "live" means here
  - Frames are read in order and a frame of video time v is not touched before wall time T0 + v (--no-wait drops the waiting, for tests;
    the order of work stays the same, so nothing ever uses a later frame either way).
  - Everything is decided from frames already seen: the race clock on screen gives the start (recognised 1 s after it starts to run)
    and the finish (recognised 1 s after it stops); a time point uses the frames at t, t-0.5 s and t-1.0 s.
  - A line is "ready" at the wall time its work ended. That time is written down and is the time the subtitle appears.
  - While the identifier works, frames pile up; they are read quickly afterwards. If the system falls behind, the dense pictures of the
    home straight are skipped first (counted in the output).

What is said
  - identifier v11 (identify_l1.py's default: fast flow, fallback, leader threshold 0.5, cap classifier caps2_a) every --step seconds
    from 9 s after the start; the lines come from fixed patterns (no language model), so writing a line takes no time.
  - at the finish: the winner call of --winner-call (most-named leader of the last 6 s, kept only when still in front in the last 2 s).
    To have it ready at once, the pictures of the home straight (5 a second, from 14 s before the finish expected from the distance)
    are run as they come.
  - no line speaks of a change ("moves up", "overtakes"): comparing two time points does not show real changes (top5/RESULT_KO.md).
  - a cap of 11 or above cannot be told apart: such a number is said only when the race card has exactly one runner of 11 or above.
  - --finish-leader (added 2026-11-23, live3/DESIGN_KO.md): from 36 s before the finish expected from the distance, the leader comes from
    the finish-stretch model learned on official passing times (identify_l1.py --v10's leader part, finish2/models/fs_leader_all.joblib).
    The finish is not known while running, so the stretch starts at expected_seconds(distance) - 36 s of race time. Statements, order and
    the winner call are unchanged. Off by default (live2/RESULT_KO.md: leader lines 58.7 %, mostly wrong 36-12 s before the finish).
Never read: the running-order strip, the official result, the sound, the distance-to-go band.

Writes OUT/live.json (every line with its times, what the identifier gave, how late the system was), OUT/live.srt and, with ffmpeg,
OUT/live_subtitled.mp4 (no sound, 1280 px wide)."""
import argparse
import json
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import identify_l1 as L

DIGITS = (118, 968, 302, 1012)          # inside the blue pill of the race clock (as code/fn2_times.py)
DENSE_STEP, DENSE_BEFORE = 0.2, 14.0    # home-straight pictures: one every 0.2 s, from 14 s before the expected finish
FIRST_POINT = 9.0                       # first identifier time point, seconds after the start (as the script demo)


def expected_seconds(d):                # as code/ls2_run.py
    return d * (0.062 if d <= 1200 else 0.0635 if d <= 1400 else 0.066)


class Clock:
    """The race clock, frame by frame (the same pixel tests as code/fn2_times.py, made to run forwards)."""

    def __init__(self):
        self.prev, self.run_start, self.last_tick, self.start, self.finish = None, None, None, None, None

    def feed(self, v, bgr):
        x1, y1, x2, y2 = L.TIMER
        hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        h, s, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        on = float(((h >= 100) & (h <= 125) & (s > 140) & (val > 90)).mean()) >= 0.06 and float(((s < 60) & (val > 200)).mean()) >= 0.10
        dx1, dy1, dx2, dy2 = DIGITS
        d = cv2.cvtColor(bgr[dy1:dy2, dx1:dx2], cv2.COLOR_BGR2HSV)
        dw = (d[..., 1] < 90) & (d[..., 2] > 170)
        tick = on and self.prev is not None and float((dw != self.prev).mean()) > 0.01
        self.prev = dw
        if tick:
            if self.last_tick is None or v - self.last_tick > 1.0:
                self.run_start = v                                   # a new run of changing digits
            self.last_tick = v
            if self.start is None and v - self.run_start >= 1.0:
                self.start = self.run_start                          # the clock has been running for 1 s: the race started at run_start
        elif self.start is not None and self.finish is None and on and v - self.last_tick > 1.0 and self.last_tick - self.start >= 20:
            self.finish = round(self.last_tick - 0.2, 3)             # the last change comes about 0.2 s after the clock stops
        return on


def josa(word, a="이", b="가"):
    c = word[-1]
    return a if "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 else b


class Lines:
    """Fixed patterns. Only what the identifier gave is said; where it gave nothing, confirmed information (race, distance, time)."""

    def __init__(self, card, names):
        self.card, self.k, self.last = card, 0, None
        self.name = {r["number"]: r["name"] for r in card["runners"]} if names else {}
        high = [r["number"] for r in card["runners"] if r["number"] >= 11]
        self.high = high[0] if len(high) == 1 else None

    def number(self, n):
        """The number as it may be said, or None (a cap of 11 or above that cannot be told apart)."""
        if n >= 11:
            n = self.high
        return n

    def label(self, n):
        return f"{n}번 {self.name[n]}" if n in self.name else f"{n}번"

    def pick(self, options):
        self.k += 1
        return options[self.k % len(options)]

    def start(self):
        c = self.card
        return f"{c['place']} {c['race']}경주 {c['distance']}m, {len(c['runners'])}두가 출발했습니다."

    def point(self, res, race_time):
        c = self.card
        lead = self.number(res["leader"]) if res.get("leader") else None
        group = [m for m in (self.number(n) for n in res["statements"]) if m and m != lead]
        key = (lead, tuple(sorted(group)))
        same, self.last = key == self.last and (lead or group), key
        gl = ", ".join(self.label(n) for n in group)
        if lead:
            ll = self.label(lead)
            if same:
                return self.pick([f"여전히 선두는 {ll}.", f"{ll}, 계속 선두입니다."]) + (f" 선두권에 {gl}." if group else "")
            head = self.pick([f"선두는 {ll}.", f"{ll}{josa(ll)} 선두입니다.", f"{ll}, 선두에 있습니다."])
            return head + (" " + self.pick([f"선두권에 {gl}.", f"{gl}도 선두권입니다."]) if group else "")
        if group:
            pair = next((o for o in res.get("order", []) if self.number(o["ahead"]) in group and self.number(o["behind"]) in group), None)
            if pair and not same:
                a, b = self.label(self.number(pair["ahead"])), self.label(self.number(pair["behind"]))
                return f"선두권에 {gl}. {a}{josa(a)} {b}보다 앞에 있습니다."
            if same:
                return self.pick([f"선두권은 그대로 {gl}.", f"{gl}, 계속 선두권입니다."])
            return self.pick([f"선두권에 {gl}{josa(gl)} 있습니다.", f"{gl}, 선두권에서 달립니다."])
        return self.pick([f"{c['place']} {c['race']}경주 {c['distance']}m, 출발 후 {int(race_time)}초가 지났습니다.", f"{c['distance']}m 경주, 출발 후 {int(race_time)}초입니다."])

    def finish(self, winner):
        c = self.card
        w = self.number(winner) if winner else None
        if w:
            lw = self.label(w)
            return f"결승선 통과! {lw}{josa(lw)} 가장 먼저 들어옵니다!"
        return f"결승선을 통과했습니다. {c['place']} {c['race']}경주가 끝났습니다."


def winner_from(dense, finish):
    """The rule of LeadingGroupIdentifier.winner_call_checked, on pictures already run (dense = [(v, numbers picked as leader, (boxes, P))])."""
    count, seen_by = Counter(), []
    for v, picks, raw in dense:
        tf = finish - v
        if not (-0.05 <= tf <= L.WINNER_SPAN + 0.05):
            continue
        count.update(picks)
        if tf <= L.WINNER_CHECK_SECONDS + 0.05:
            seen_by.append(raw)
    top = count.most_common(1)
    cand = top[0][0] if top and top[0][1] >= L.WINNER_NEED else None
    info = {"candidate": cand, "pictures_naming_each": dict(count), "seen": 0, "front": 0,
            "pictures_in_the_last_6s": sum(1 for v, _, _ in dense if -0.05 <= finish - v <= L.WINNER_SPAN + 0.05)}
    if cand is None:
        return None, info
    cap = min(cand, 11) - 1
    for boxes, P in seen_by:
        if P is None or not len(boxes):
            continue
        b = np.array(boxes, float)
        cx, h = (b[:, 0] + b[:, 2]) / 2, b[:, 3] - b[:, 1]
        i = int(np.argmax(P[:, cap]))
        if P[i, cap] < L.WINNER_SEE:
            continue
        info["seen"] += 1
        info["front"] += int(int(((cx - cx[i]) > L.WINNER_AHEAD * h[i]).sum()) == 0)
    return (cand if info["seen"] >= 1 and info["front"] >= 0.5 * info["seen"] else None), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--racecard", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step", type=float, default=3.0)
    ap.add_argument("--no-wait", action="store_true", help="do not wait for the wall clock (a test of the code, not a live run)")
    ap.add_argument("--no-names", action="store_true", help="say numbers only")
    ap.add_argument("--no-video", action="store_true", help="do not write the subtitled video")
    ap.add_argument("--finish-leader", action="store_true", help="leader from the finish-stretch model in the last 36 s expected from the distance")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    card = json.loads(Path(a.racecard).read_text())
    lines = Lines(card, not a.no_names)
    L.wait_gpu()
    ident = L.LeadingGroupIdentifier("sep06", fast=True, fallback=True, tau_leader=L.TAU_LEADER_V9, cap_weights=L.CNN_WEIGHTS_V11,
                                    finish_leader=a.finish_leader)
    cap = cv2.VideoCapture(a.video)
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (1920, 1080), "the graphics positions assume a 1920x1080 broadcast"
    ok, first = cap.read()
    for _ in range(3):
        ident.picture(first)                   # warm the models up before the clock starts; the result is thrown away
    ident.history.clear()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    clock, buf, events, dense = Clock(), deque(), [], []
    expect = expected_seconds(card["distance"])
    next_point, next_dense, started_line, done = FIRST_POINT, None, False, None
    late_max, dense_skipped, frames = 0.0, 0, 0
    T0 = time.perf_counter()
    now = lambda: time.perf_counter() - T0          # wall time on the video's own scale

    def emit(kind, v, line, **more):
        events.append({"kind": kind, "video_time": round(v, 3), "race_time": None if clock.start is None else round(v - clock.start, 2),
                       "ready": round(max(now(), v) if not a.no_wait else v + more.get("work_seconds", 0.0), 3), "line": line, **more})
        e = events[-1]
        print(f"{e['video_time']:7.2f} -> {e['ready']:7.2f}  [{kind}] {line}", flush=True)

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        v = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frames += 1
        if not a.no_wait and now() < v:
            time.sleep(v - now())
        late = 0.0 if a.no_wait else max(0.0, now() - v)
        late_max = max(late_max, late)
        clock.feed(v, bgr)
        buf.append((v, bgr))
        while buf and v - buf[0][0] > 1.15:
            buf.popleft()
        if clock.start is None:
            continue
        r = v - clock.start
        if not started_line:
            started_line = True
            emit("start", v, lines.start(), late_at_start=round(late, 3))
        if clock.finish is not None and done is None:
            w0 = time.perf_counter()
            winner, info = winner_from(dense, clock.finish)
            emit("finish", v, lines.finish(winner), winner=winner, winner_call=info, finish_video_time=clock.finish,
                 seconds_after_the_finish=None, work_seconds=round(time.perf_counter() - w0, 3))
            events[-1]["seconds_after_the_finish"] = round(events[-1]["ready"] - clock.finish, 2)
            done = v
        if done is not None:
            if v - done > 3.0:
                break
            continue
        if r >= next_point:
            w0 = time.perf_counter()
            near = lambda t: min(buf, key=lambda f: abs(f[0] - t))[1]
            stretch = a.finish_leader and r >= expect - L.LATE_SECONDS
            res = ident.point(v, [bgr, near(v - 0.5), near(v - 1.0)], late=stretch)
            work = time.perf_counter() - w0
            emit("point", v, lines.point(res, r), statements=res["statements"], leader=res.get("leader"), leader_source=res.get("leader_source"),
                 order=res.get("order", []), late_at_start=round(late, 3), work_seconds=round(work, 3),
                 **({"finish_stretch": stretch, "leader_strip_model": res.get("leader_strip_model", res.get("leader"))} if a.finish_leader else {}))
            next_point += a.step
            continue
        if r >= expect - DENSE_BEFORE:
            if next_dense is None:
                next_dense = v
            if v >= next_dense:
                next_dense += DENSE_STEP
                if late > 0.15:
                    dense_skipped += 1                      # behind: let this picture go
                    continue
                riders = ident.picture(bgr)[0]
                dense.append((v, [x["number"] for x in riders if x.get("leader_pick")], ident.last_raw))
    cap.release()

    for k, e in enumerate(events):
        e["shown_until"] = round(min(events[k + 1]["ready"] if k + 1 < len(events) else e["ready"] + 6, e["ready"] + 6) - 0.05, 3)
        e["delay"] = round(e["ready"] - e["video_time"], 3)
    pts_delay = sorted(e["delay"] for e in events if e["kind"] == "point")
    res = {"video": a.video, "racecard": card, "configuration": "v11 (F2+caps2)" + (" + finish-stretch leader" if a.finish_leader else "") + " + winner call, pattern lines", "waited_for_the_wall_clock": not a.no_wait,
           "clock": {"start": clock.start, "finish": clock.finish, "expected_seconds": round(expect, 1)}, "frames_read": frames,
           "lines": len(events), "delay_seconds": {"median": pts_delay[len(pts_delay) // 2] if pts_delay else None, "max": pts_delay[-1] if pts_delay else None},
           "furthest_behind_seconds": round(late_max, 3), "dense_pictures": {"run": len(dense), "skipped_when_behind": dense_skipped}, "events": events}
    (out / "live.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))

    def stamp(t):
        return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}".replace(".", ",")

    (out / "live.srt").write_text("\n".join(f"{k + 1}\n{stamp(e['ready'])} --> {stamp(e['shown_until'])}\n{e['line']}\n" for k, e in enumerate(events)))
    print({k: res[k] for k in ("clock", "lines", "delay_seconds", "furthest_behind_seconds", "dense_pictures")}, flush=True)
    if a.no_video:
        return
    d = res["delay_seconds"]["median"]
    label = (f"화면만 보고 실시간으로 낸 자막 · 식별기 v11 + 문장 틀 · 문장마다 약 {d:.1f}초 뒤에 표시" if not a.no_wait
             else "화면만 보고 낸 자막 · 식별기 v11 + 문장 틀 · 기다리지 않고 돌린 시험")
    style = "FontName=Noto Sans CJK KR,FontSize=22,PrimaryColour=&H00FFFFFF,BackColour=&H90000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=28"
    vf = (f"scale=1280:-2,subtitles={out / 'live.srt'}:force_style='{style}',"
          f"drawtext=fontfile=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc:text='{label}':x=16:y=14:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.video, "-vf", vf, "-an", "-r", "30", "-c:v", "libx264", "-crf", "24", "-preset", "fast",
                    str(out / "live_subtitled.mp4")], check=True)
    print("written", out / "live_subtitled.mp4", flush=True)


if __name__ == "__main__":
    main()
