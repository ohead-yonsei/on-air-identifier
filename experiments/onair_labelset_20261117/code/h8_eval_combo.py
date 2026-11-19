"""caps3: combinations of a cap classifier and a set of position models, compared with v11 on the same riders (same boxes).
usage: h8_eval_combo.py OUT.json CONFIG [CONFIG ...] -- DAY [DAY ...]
       CONFIG = LABEL:CAPS:MODELS     CAPS = name under caps2/dense; MODELS = "old" (the models in use) or a name under caps2/models (h7_train_models.py)
The first comparison is always against v11 = A:old (classifier caps2_a.pt with the models in use).
Metrics and time points as h5_eval_caps.py (official order as the answer; races whose clock agrees; v10's leader rule).

Rule for choosing among the candidates on days seen before (written before the first run):
  no harm   leader wrong share at most 1.0 point above v11's, A and B at most 1.0 point below v11's
  choice    among the candidates without harm, the highest "leader right share"
  go on     the choice goes to new days only if its leader right share or its B is above v11's with the 95 % interval above 0"""
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import fn_common as fc
import h5_eval_caps as h5

OLD = {"order": ROOT / "night" / "l1_fast.joblib", "strip": ROOT / "night" / "leader_fast.joblib", "fs": ROOT / "finish2" / "models" / "fs_leader_all.joblib"}


def models(name):
    if name == "old":
        return {k: joblib.load(v) for k, v in OLD.items()}
    d = ROOT / "caps2" / "models"
    return {"order": joblib.load(d / f"l1_{name}.joblib"), "strip": joblib.load(d / f"leader_{name}.joblib"), "fs": joblib.load(d / f"fs_leader_{name}.joblib")}


def run(caps, mods, days):
    m = models(mods)
    h5.ORDER, h5.STRIP, h5.FS = m["order"], m["strip"], m["fs"]
    lead, ab, win = {}, {}, []
    for d in days:
        a, data = h5.leader_rows(caps, d, 0.7)
        lead.update(a)
        ab.update(h5.statement_rows(data, d))
        win += h5.winner(caps, d)
    return {"lead": lead, "ab": ab, "win": win}


def main():
    args = sys.argv[1:]
    out = Path(args.pop(0))
    k = args.index("--")
    configs, days = [c.split(":") for c in args[:k]], args[k + 1:]
    for d in days:
        fc.FOLDER.setdefault(d, d)
    flat = lambda by: [r for v in by.values() for r in v]
    weak = lambda by: {key: [r for r in v if r["true"] in h5.WEAK] for key, v in by.items()}
    base = run("A", "old", days)
    res = {"days": days, "races": len(base["lead"]), "configs": {}}

    def show(label, r, ref=None):
        called = [w for w in r["win"] if w["call"] is not None]
        v = {"leader": h5.lead_sum(flat(r["lead"])), "numbers": h5.ab_sum(flat(r["ab"])), "true_leader_7_10_11+": h5.lead_sum(flat(weak(r["lead"]))),
             "winner": {"races": len(r["win"]), "called": len(called), "right": sum(w["rank"] == 1 for w in called), "wrong_ranks": sorted(w["rank"] or 99 for w in called if w["rank"] != 1)}}
        if ref is not None:
            v["difference"] = {"readable": h5.diff(r["lead"], ref["lead"], h5.lead_sum, "readable"), "leader_right": h5.diff(r["lead"], ref["lead"], h5.lead_sum, "right"),
                               "leader_wrong": h5.diff(r["lead"], ref["lead"], h5.lead_sum, "wrong"), "A": h5.diff(r["ab"], ref["ab"], h5.ab_sum, "A"), "B": h5.diff(r["ab"], ref["ab"], h5.ab_sum, "B")}
        res["configs"][label] = v
        print(f"{label:12s} readable {v['leader']['readable']}% | leader right {v['leader']['right']}% wrong {v['leader']['wrong']}% | A {v['numbers']['A']}% ({v['numbers']['right']}/{v['numbers']['said']}) B {v['numbers']['B']}%"
              f" | weak caps readable {v['true_leader_7_10_11+']['readable']}% right {v['true_leader_7_10_11+']['right']}% | winner {v['winner']['called']} called, {v['winner']['right']} right {v['winner']['wrong_ranks']}"
              + ("" if ref is None else f"\n{'':12s} minus v11: {json.dumps(v['difference'])}"), flush=True)
        return v

    print(f"{res['races']} races, {len(flat(base['lead']))} time points on {days}")
    b = show("v11 (A:old)", base)
    ok = {}
    for label, caps, mods in configs:
        v = show(label, run(caps, mods, days), base)
        harm = v["leader"]["wrong"] - b["leader"]["wrong"] > 1.0 or v["numbers"]["A"] - b["numbers"]["A"] < -1.0 or v["numbers"]["B"] - b["numbers"]["B"] < -1.0
        v["no_harm"] = not harm
        if not harm:
            ok[label] = v["leader"]["right"]
    choice = max(ok, key=ok.get) if ok else None
    res["choice"] = choice
    if choice:
        d = res["configs"][choice]["difference"]
        res["goes_to_new_days"] = bool(d["leader_right"][0] > 0 or d["B"][0] > 0)
    else:
        res["goes_to_new_days"] = False
    print("choice:", choice, "| goes to new days:", res["goes_to_new_days"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
