#!/bin/bash
# caps2: the identifier with the new cap classifier on every strip frame of the six benchmark test days (v9 flow, nothing else changed).
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=caps2/bench.log
mkdir -p caps2/bench_ident
echo "[$(date -Is)] start" >> $L
for d in otherday6 otherday7 otherday8 otherday9 otherday10 otherday11; do
  $PY identifier/identify_l1.py --day-dir $d --cap-weights cnn/models/caps2_a.pt --out caps2/bench_ident/CAPS2_$d.json > caps2/bench_ident/CAPS2_$d.log 2>&1
  echo "[$(date -Is)] $d exit $?: $(tail -1 caps2/bench_ident/CAPS2_$d.log)" >> $L
done
echo "[$(date -Is)] done" >> $L
