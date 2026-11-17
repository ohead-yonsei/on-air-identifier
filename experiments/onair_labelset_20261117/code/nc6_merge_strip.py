"""Copy live_commentary/DAY/results_f2/R??.json to results_f2s/, adding to each time point the strip read in results_order (joined by time).
usage: nc6_merge_strip.py DAY     Standard library only. results_f2 was run without reading the strip; the strip is needed only for scoring."""
import json
import sys
from pathlib import Path

O = Path(__file__).resolve().parent.parent / "live_commentary" / sys.argv[1]
(O / "results_f2s").mkdir(exist_ok=True)
n = with_strip = 0
for f in sorted((O / "results_f2").glob("R??.json")):
    d = json.loads(f.read_text())
    ref = {round(p["t"], 2): p for p in json.loads((O / "results_order" / f.name).read_text())["points"]}
    for p in d["points"]:
        r = ref[round(p["t"], 2)]
        p["strip"], p["strip_on_screen"], p["validated_segment"] = r.get("strip"), r.get("strip_on_screen"), r.get("validated_segment")
        n += 1
        with_strip += bool(r.get("strip"))
    (O / "results_f2s" / f.name).write_text(json.dumps(d, indent=1))
print("time points", n, "with the strip", with_strip)
