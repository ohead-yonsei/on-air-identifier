"""finish3, step 1: official results for more days, with the unchanged parser of fn1_official.py.
usage: g1_official_days.py RAW_DIR DAY=YYYY-MM-DD [DAY=YYYY-MM-DD ...]      writes finish2/official/DAY.json (where fn_common.official reads)
Numbers only; the raw pages go to RAW_DIR outside the project."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn1_official as f1

ROOT = Path(__file__).resolve().parent.parent
raw = sys.argv[1]
f1.DAYS = dict(a.split("=") for a in sys.argv[2:])
for day in f1.DAYS:
    assert not (ROOT / "finish2" / "official" / f"{day}.json").exists(), f"{day} is already there"
sys.argv = [sys.argv[0], str(ROOT / "finish2" / "official"), raw]
f1.main()
