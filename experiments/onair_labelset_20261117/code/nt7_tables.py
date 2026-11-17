"""Print the overnight score (night/score.json) as Markdown tables for RESULT_KO.md. usage: python3 nt7_tables.py night/score.json
Standard library only."""
import json
import sys

s = json.load(open(sys.argv[1]))
DAY = {"otherday8": "9/19", "otherday9": "9/12", "otherday10": "9/5", "otherday11": "8/30"}
P, D, IV, V = s["pooled"], s["by_day"], s["intervals"], s["verdict"]


def pc(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def iv(k):
    a, b = IV[k]
    return f"{a * 100:+.1f} ~ {b * 100:+.1f}%p" if "minus" in k else f"{a * 100:.1f} ~ {b * 100:.1f}%"


days = s["days"]
print(f"races {s['races']}, points {P['L1']['points']}\n")
print("| 구성 | " + " | ".join(f"{DAY.get(d, d)} A / B" for d in days) + " | 합계 A | 합계 B |")
print("|---|" + "---|" * (len(days) + 2))
for k in ("T1", "L1", "L1h", "L1s", "L1x", "F1", "F1x"):
    print(f"| {k} | " + " | ".join(f"{pc(D[d][k]['A'])} / {pc(D[d][k]['B'])}" for d in days) + f" | {pc(P[k]['A'])} | {pc(P[k]['B'])} |")
print("\npoints per day:", {DAY.get(d, d): D[d]["L1"]["points"] for d in days}, "statements L1:", P["L1"]["statements"])
print("\n| 선두 규칙 | " + " | ".join(DAY.get(d, d) for d in days) + " | 합계 정확도 | 맞게 말한 시점 |")
print("|---|" + "---|" * (len(days) + 2))
for k in ("HAND", "LEAD", "LEAD50", "LEADx", "F_LEAD"):
    print(f"| {k} | " + " | ".join(f"{D[d][k]['right']}/{D[d][k]['statements']} ({pc(D[d][k]['precision'])})" for d in days)
          + f" | {P[k]['right']}/{P[k]['statements']} ({pc(P[k]['precision'])}) | {pc(P[k]['coverage'])} |")
print("\n| 앞뒤 규칙 | " + " | ".join(DAY.get(d, d) for d in days) + " | 합계 |")
print("|---|" + "---|" * (len(days) + 1))
for k in ("H0", "PAIR", "PAIRx", "F_PAIR"):
    print(f"| {k} | " + " | ".join(f"{D[d][k]['right']}/{D[d][k]['scoreable']} ({pc(D[d][k]['right_share'])})" for d in days) + f" | {P[k]['right']}/{P[k]['scoreable']} ({pc(P[k]['right_share'])}) |")
print("\nintervals:")
for k in IV:
    print(f"  {k}: {iv(k)}")
print("\nverdicts:")
for k, v in V.items():
    print(f"  {k}: {v}")
