"""finish3 stage 2, exploration: g5_explore.py again with a looser answer while training. In g5 a rider counts as "first" only when no other
horse is within 3 m (few such moments in the pack: 3.7 % of the samples). Here every rider whose number is first counts as "first", and
riders more than 3 m behind the first as "not first" (the others are still left out). Everything else as g5_explore.py.
usage: g5b_explore_labels.py OUT.json"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g5_explore as g5


def samples(data, days):
    X, y = [], []
    for d in days:
        for rr in data[d].values():
            if not rr["agree"]:
                continue
            for e in rr["pics"].values():
                ref = e["ref"]
                if ref is None:
                    continue
                for num, x in zip(e["nums"], e["X"]):
                    if num == ref["leader"] or num not in ref["leader_lenient"]:
                        X.append(x)
                        y.append(int(num == ref["leader"]))
    return np.array(X), np.array(y)


g5.samples = samples
_dump = g5.joblib.dump
g5.joblib.dump = lambda m, p: _dump(m, Path(str(p).replace("_all.joblib", "b_all.joblib")))
g5.main()
