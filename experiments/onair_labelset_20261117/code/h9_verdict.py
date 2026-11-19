"""caps3: the two verdicts of caps3/DESIGN_KO.md, read from the output of h8_eval_combo.py on the new days.
usage: h9_verdict.py EVAL.json         (standard library only)"""
import json
import sys

d = json.load(open(sys.argv[1]))
c, base = d["configs"], d["configs"]["v11 (A:old)"]


def delta(v, part, key):
    return round(v[part][key] - base[part][key], 1)


out = {}
for label, wrong_limit, need in (("B", 1.0, "either"), ("B+M", 2.0, "leader")):
    v = c[label]
    dw, da, db, dr = delta(v, "leader", "wrong"), delta(v, "numbers", "A"), delta(v, "numbers", "B"), delta(v, "leader", "right")
    no_harm = dw <= wrong_limit and da >= -1.0 and db >= -1.0
    iv = v["difference"]
    gain = iv["leader_right"][0] > 0 or (need == "either" and iv["B"][0] > 0)
    out[label] = {"leader_right": dr, "leader_right_interval": iv["leader_right"], "leader_wrong": dw, "leader_wrong_interval": iv["leader_wrong"], "A": da, "A_interval": iv["A"],
                  "B": db, "B_interval": iv["B"], "no_harm": bool(no_harm), "gain": bool(gain), "confirmed": bool(no_harm and gain)}
    print(f"{label}: leader right {dr:+.1f} {iv['leader_right']}, leader wrong {dw:+.1f} {iv['leader_wrong']}, A {da:+.1f} {iv['A']}, B {db:+.1f} {iv['B']} -> no harm {no_harm}, gain {gain}, CONFIRMED {no_harm and gain}")
d["verdicts"] = out
json.dump(d, open(sys.argv[1], "w"), indent=1)
