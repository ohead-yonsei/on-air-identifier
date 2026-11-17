"""Which boxes to speak about, and how strict 'sure' must be: trade-off between criteria A and B.

usage: python q1_statement_rules.py dev|test
A statement = "number N is in the leading group", made for a rider among the front K boxes (vertical rule)
whose answer is sure and whose learned-classifier probability for that number is at least T (T=0: plain sure).
Uses stripset/out/D0 only. No model calls. Writes order/statement_<split>.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "stripset"
split = sys.argv[1]
frames = [m for m in json.loads((S / "manifest.json").read_text())["frames"] if m["split"] == split]
KS, TS = (1, 2, 3), (0.0, 0.1, 0.3, 0.5, 0.7)
res = {}
for K in KS:
    for T in TS:
        st = ok3 = ok5 = fr_any = fr_ok = 0
        for m in frames:
            riders = json.loads((S / "out" / "D0" / f"{m['frame']}.json").read_text())["riders"]
            B = m["strip"]
            front = sorted(riders, key=lambda r: -r["box"][3])[:K]
            said = [r["no"] for r in front if r["state"] == "sure" and (r["p_learned"] or 0) >= T]
            st += len(said)
            ok3 += sum(n in B[:3] for n in said)
            ok5 += sum(n in B for n in said)
            fr_any += bool(said)
            fr_ok += any(n in B[:3] for n in said)
        res[f"K{K}_T{T}"] = {"K": K, "T": T, "statements": st, "in_top3": ok3, "in_top5": ok5, "A": ok3 / st if st else 0,
                             "A_top5": ok5 / st if st else 0, "frames_with_statement": fr_any / len(frames), "B": fr_ok / len(frames)}
(ROOT / "order" / f"statement_{split}.json").write_text(json.dumps({"split": split, "frames": len(frames), "configs": res}, indent=1))
print(f"split {split}: {len(frames)} frames")
print("K  T     statements  A=in top3      in top5   frames with a statement   B=frames with a correct one")
for k, v in res.items():
    print(f"{v['K']}  {v['T']:.1f}   {v['statements']:4d}        {v['in_top3']:3d} ({v['A']:.3f})   {v['A_top5']:.3f}     {v['frames_with_statement']:.3f}                     {v['B']:.3f}")
