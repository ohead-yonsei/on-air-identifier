"""Score leader claims: A (picture only), B (picture + confirmed numbers by prompt), G (gated in code). See commentary2/DESIGN_KO.md.
G = B's answer where the identifier confirmed at least one number, no claim otherwise."""
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "commentary2"
rows = {}
for line in (C / "raw.jsonl").read_text().splitlines():
    r = json.loads(line)
    if not r["failure"]:
        rows[(r["key"], r["condition"])] = r
keys = sorted({k for k, _ in rows})
complete = [k for k in keys if (k, "A") in rows and (k, "B") in rows]


def claims(k, cond):
    if cond == "G":
        r = rows[(k, "B")]
        return list(dict.fromkeys(r["leaders"])) if r["facts"] else []
    return list(dict.fromkeys(rows[(k, cond)]["leaders"]))


out = {"frames_complete": len(complete), "failures_or_missing": len(keys) - len(complete)}
for kind in ("strip", "finish"):
    ks = [k for k in complete if rows[(k, "A")]["kind"] == kind]
    res, wrong = {}, {}
    for cond in ("A", "B", "G"):
        w = {}
        t = {"claims": 0, "wrong": 0, "frames_with_wrong": 0, "frames_with_right": 0, "frames_silent": 0}
        for k in ks:
            ref = rows[(k, "A")]["reference"]
            said = claims(k, cond)
            w[k] = sum(n not in ref for n in said)
            t["claims"] += len(said)
            t["wrong"] += w[k]
            t["frames_with_wrong"] += w[k] > 0
            t["frames_with_right"] += any(n in ref for n in said)
            t["frames_silent"] += not said
        t["wrong_rate"] = t["wrong"] / t["claims"] if t["claims"] else None
        res[cond], wrong[cond] = t, w

    def paired(a, b):
        fewer = sum(wrong[b][k] < wrong[a][k] for k in ks)
        more = sum(wrong[b][k] > wrong[a][k] for k in ks)
        n, m = fewer + more, min(fewer, more)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0
        return {"frames_with_fewer_wrong": fewer, "frames_with_more_wrong": more, "sign_test_p": p}

    facts_n = sum(len(rows[(k, "B")]["facts"]) for k in ks)
    facts_ok = sum(n in rows[(k, "B")]["reference"] for k in ks for n in rows[(k, "B")]["facts"])
    ag, bg = paired("A", "G"), paired("B", "G")
    nofact = [k for k in ks if not rows[(k, "B")]["facts"]]
    out[kind] = {"frames": len(ks), "conditions": res, "A_vs_G": ag, "B_vs_G": bg,
                 "supported_A_vs_G": bool(ag["frames_with_fewer_wrong"] > ag["frames_with_more_wrong"] and ag["sign_test_p"] < 0.05),
                 "identifier_facts": {"frames_with_facts": len(ks) - len(nofact), "facts": facts_n, "facts_right": facts_ok},
                 "frames_without_facts": {"frames": len(nofact), "B_claims": sum(len(claims(k, "B")) for k in nofact), "B_wrong": sum(wrong["B"][k] for k in nofact),
                                          "A_claims": sum(len(claims(k, "A")) for k in nofact), "A_wrong": sum(wrong["A"][k] for k in nofact)}}
    print(f"== {kind}: {len(ks)} frames | identifier facts {facts_ok}/{facts_n} right, in {len(ks) - len(nofact)} frames")
    for cond in ("A", "B", "G"):
        t = res[cond]
        print(f"  {cond}: claims {t['claims']:3d} wrong {t['wrong']:3d} | frames with a wrong claim {t['frames_with_wrong']:3d}, with a right claim {t['frames_with_right']:3d}, silent {t['frames_silent']:3d}")
    print("  A vs G:", ag, "| B vs G:", bg)
    print("  frames without identifier facts:", out[kind]["frames_without_facts"])
(C / "score.json").write_text(json.dumps(out, indent=1))
