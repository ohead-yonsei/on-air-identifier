"""finish3 stage 2: the pictures of the last 37 s, 5 a second, for a day that fn_common does not list (fn6_dense.py unchanged).
usage: g10_dense_last.py DAY        writes finish2/dense/DAY.json, where fn9_winner.py reads"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fn_common as fc

fc.FOLDER.setdefault(sys.argv[1], sys.argv[1])
import fn6_dense

fn6_dense.main()
