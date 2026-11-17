"""Score the 27B identifier against the 9B identifier on held-out frames. See models/DESIGN_IDENTIFIER_KO.md."""
import json
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "stripset"
frames = [m for m in json.loads((S / "manifest.json").read_text())["frames"] if m["split"] == "test"]
DIRS = {"9B": (S / "out" / "D0", S / "out_past" / "D0"), "27B": (S / "out27" / "D0", S / "out_past27" / "D0")}


def statements(frame, which):
    cur, past = DIRS[which]
    votes, ms, sure, riders_n = Counter(), 0.0, 0, 0
    for p in (cur / f"{frame}.json", past / f"{frame}_m05.json", past / f"{frame}_m10.json"):
        d = json.loads(p.read_text())
        ms += d["total_ms"]
        riders_n += len(d["riders"])
        sure += sum(r["state"] == "sure" for r in d["riders"])
        front = sorted(d["riders"], key=lambda r: -r["box"][3])[:2]
        for n in {r["no"] for r in front if r["state"] == "sure"}:
            votes[n] += 1
    return [n for n, v in votes.items() if v >= 2], ms, sure, riders_n


res, good = {}, {}
for which in DIRS:
    st = ok = ok5 = any_ = 0
    ms = sure = rn = 0
    good[which] = {}
    for m in frames:
        said, t, s, r = statements(m["frame"], which)
        B = m["strip"]
        st += len(said)
        ok += sum(n in B[:3] for n in said)
        ok5 += sum(n in B for n in said)
        any_ += bool(said)
        good[which][m["frame"]] = any(n in B[:3] for n in said)
        ms += t
        sure += s
        rn += r
    n = len(frames)
    res[which] = {"statements": st, "in_top3": ok, "A": ok / st if st else 0.0, "A_top5": ok5 / st if st else 0.0, "frames_with_statement": any_ / n,
                  "B": sum(good[which].values()) / n, "ms_per_time_point": ms / n, "sure_share_of_riders": sure / rn if rn else 0.0}
imp = sum(good["27B"][f] and not good["9B"][f] for f in good["9B"])
wor = sum(good["9B"][f] and not good["27B"][f] for f in good["9B"])
k, mm = imp + wor, min(imp, wor)
p = min(1.0, 2 * sum(comb(k, i) for i in range(mm + 1)) / 2 ** k) if k else 1.0
better = bool(imp > wor and p < 0.05 and res["27B"]["A"] >= res["9B"]["A"] - 0.02)
out = {"frames": len(frames), "results": res, "improved_frames": imp, "worsened_frames": wor, "sign_test_p": p, "better": better}
(ROOT / "models" / "score_identifier.json").write_text(json.dumps(out, indent=1))
print("held-out frames:", len(frames))
for which, v in res.items():
    print(f"  {which:4s} statements {v['statements']:3d}  A {v['A']:.3f} (top5 {v['A_top5']:.3f})  frames with statement {v['frames_with_statement']:.3f}  B {v['B']:.3f}  "
          f"sure share of riders {v['sure_share_of_riders']:.3f}  {v['ms_per_time_point'] / 1000:.1f} s per time point")
print(f"frames: improved {imp} / worsened {wor}  p={p:.4f} | A change {res['27B']['A'] - res['9B']['A']:+.3f} | better: {better}")
