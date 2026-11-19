import sys
from collections import Counter, defaultdict
sys.path.insert(0, "code")
import numpy as np
import fn_common as fc, w4_winner_guard as w4
days = "20261115 otherday otherday2 otherday3 otherday4 otherday5 otherday6 otherday7 otherday8 otherday9 otherday10 otherday11 otherday14 otherday15 otherday16 otherday17 otherday18 otherday19 otherday20 otherday21 otherday22".split()
for T in (0.4, 1.0):
    for margin in (0.0, 0.2, 0.4):
        k, conf, races = Counter(), defaultdict(Counter), 0
        for d in days:
            fc.FOLDER.setdefault(d, d); off = fc.official(d)
            for race, dr in sorted(w4.dense(d).items()):
                if not dr["clock_agrees"]: continue
                hs = sorted((h for h in off[race]["horses"] if h.get("rank") and h.get("time", {}).get("finish")), key=lambda h: h["rank"])
                if len(hs) < 2 or hs[1]["time"]["finish"] - hs[0]["time"]["finish"] < margin: continue
                races += 1
                wcap = min(hs[0]["number"], 11) - 1
                for p in dr["pictures"]:
                    if not (-0.05 <= p["to_finish"] <= T + 0.05) or not p["boxes"]: continue
                    b, P = np.array(p["boxes"], float), np.array(p["probs"], float)
                    i = int(np.argmax((b[:, 0] + b[:, 2]) / 2)); c = int(np.argmax(P[i])); pr = P[i, c]
                    k["pictures"] += 1
                    if pr >= 0.9:
                        k["confident"] += 1; k["confident and the winner's cap"] += c == wcap
                        if c != wcap: conf[wcap + 1][c + 1] += 1
                    elif pr >= 0.7: k["sure, not confident"] += 1; k["sure and the winner's cap"] += c == wcap
                    else: k["not sure"] += 1; k["not sure, top is the winner's cap"] += c == wcap
        print(f"last {T} s, margin >= {margin} s: {races} races, {dict(k)}  -> confident reads that are the winner's cap: {100 * k['confident and the winner\'s cap'] / max(k['confident'], 1):.1f}%")
        if T == 0.4 and margin == 0.2:
            print("   confident but another cap (winner's cap -> what was read):", {w: dict(v.most_common(4)) for w, v in sorted(conf.items())})
