"""Score detection variants against the running-order strip. See stripset/DESIGN_KO.md.

usage: python m1_score_stripset.py dev|test D0 D1 D2 D3
Reads stripset/manifest.json and stripset/out/<variant>/<frame>.json. Writes stripset/score_<split>.json.
"""
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "stripset"
split, variants = sys.argv[1], sys.argv[2:]
frames = [m for m in json.loads((S / "manifest.json").read_text())["frames"] if m["split"] == split]


def load(variant, frame):
    p = S / "out" / variant / f"{frame}.json"
    return json.loads(p.read_text()) if p.exists() else None


def score(variant):
    k = {"frames": 0, "hit1": 0, "hit2": 0, "hit3": 0, "sure": 0, "sure_in_top5": 0, "sure_in_top3": 0, "riders": 0, "ms": 0.0, "qwen_failures": 0,
         "front_sure": 0, "front_sure_is_1": 0, "front_sure_in_top3": 0, "front3_sure": 0, "front3_sure_in_top3": 0, "front3_sure_in_top5": 0,
         "frames_with_sure_top3": 0}
    hits = {}
    for m in frames:
        d = load(variant, m["frame"])
        if d is None:
            continue
        B = m["strip"]
        riders = d["riders"]
        sure = [r["no"] for r in riders if r["state"] == "sure"]
        k["frames"] += 1
        k["riders"] += len(riders)
        k["ms"] += d["total_ms"]
        k["qwen_failures"] += d["qwen_failures"]
        k["sure"] += len(sure)
        k["sure_in_top5"] += sum(n in B for n in sure)
        k["sure_in_top3"] += sum(n in B[:3] for n in sure)
        k["frames_with_sure_top3"] += any(n in B[:3] for n in sure)
        for i in range(3):
            h = B[i] in sure
            k[f"hit{i + 1}"] += h
            hits[(m["frame"], i)] = h
        front = sorted(riders, key=lambda r: -r["box"][3])[:3]
        if front and front[0]["state"] == "sure":
            k["front_sure"] += 1
            k["front_sure_is_1"] += front[0]["no"] == B[0]
            k["front_sure_in_top3"] += front[0]["no"] in B[:3]
        for r in front:
            if r["state"] == "sure":
                k["front3_sure"] += 1
                k["front3_sure_in_top3"] += r["no"] in B[:3]
                k["front3_sure_in_top5"] += r["no"] in B
    n = max(1, k["frames"])
    k["primary_top3_named_sure"] = (k["hit1"] + k["hit2"] + k["hit3"]) / (3 * n)
    k["leader_named_sure"] = k["hit1"] / n
    k["sure_in_top5_rate"] = k["sure_in_top5"] / max(1, k["sure"])
    k["riders_per_frame"] = k["riders"] / n
    k["sure_per_frame"] = k["sure"] / n
    k["ms_per_frame"] = k["ms"] / n
    return k, hits


res, allhits = {}, {}
for v in variants:
    res[v], allhits[v] = score(v)


def paired(a, b):
    keys = set(allhits[a]) & set(allhits[b])
    imp = sum(allhits[b][x] and not allhits[a][x] for x in keys)
    wor = sum(allhits[a][x] and not allhits[b][x] for x in keys)
    n, m = imp + wor, min(imp, wor)
    return {"improved": imp, "worsened": wor, "sign_test_p": min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0}


out = {"split": split, "frames_in_split": len(frames), "variants": res,
       "paired_vs_D0": {v: paired("D0", v) for v in variants if v != "D0" and "D0" in variants}}
if split == "dev" and "D0" in variants:
    base = res["D0"]["sure_in_top5_rate"]
    ok = [v for v in variants if res[v]["sure_in_top5_rate"] >= base - 0.05]
    out["chosen"] = max(ok, key=lambda v: (round(res[v]["primary_top3_named_sure"], 6), -variants.index(v)))
(S / f"score_{split}.json").write_text(json.dumps(out, indent=1))
print(f"split {split}: {len(frames)} frames")
print("var  frames  top3-named-sure  #1-named-sure  frames-with-a-sure-top3  sure-in-top5  riders/f  sure/f  ms/f  front-sure(=#1, in top3)  front3-sure in top3/top5")
for v in variants:
    k = res[v]
    n = max(1, k["frames"])
    print(f"{v}   {k['frames']:4d}   {k['primary_top3_named_sure']:.3f}            {k['leader_named_sure']:.3f}          {k['frames_with_sure_top3'] / n:.3f}                    "
          f"{k['sure_in_top5_rate']:.3f}         {k['riders_per_frame']:.1f}      {k['sure_per_frame']:.1f}    {k['ms_per_frame']:.0f}  "
          f"{k['front_sure']} ({k['front_sure_is_1']}, {k['front_sure_in_top3']})   {k['front3_sure_in_top3']}/{k['front3_sure_in_top5']} of {k['front3_sure']}")
for v, p in out["paired_vs_D0"].items():
    print(f"  D0 -> {v}: improved {p['improved']} / worsened {p['worsened']}  p={p['sign_test_p']:.4f}")
if "chosen" in out:
    print("chosen on dev:", out["chosen"])
print("qwen failures:", {v: res[v]["qwen_failures"] for v in variants})
