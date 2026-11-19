import json, sys
d = json.load(open("winner/explore/look_A.json"))
for key in sys.argv[1:]:
    r = d[key]; print("==", key, "winner", r["winner"], "second", r["second"], "call", r["call"])
    for row in sorted(r["rows"], key=lambda x: -x["tf"]):
        if row["tf"] > 4.05 or round(row["tf"] * 10) % 4: continue
        geo = sorted(((b[0] + b[2]) / 2, round(b[3] - b[1]), n, p) for b, (n, p) in zip(row["boxes"], row["who"]))
        print("  tf %.1f riders %d %s pick %s first %s fs %s | x,h,num,p: " % (row["tf"], row["riders"], row["detection"], row["pick"], row["first"], row["first_fs"])
              + " ".join("%d,%d,%d,%.2f" % (x, h, n, p) for x, h, n, p in geo))
