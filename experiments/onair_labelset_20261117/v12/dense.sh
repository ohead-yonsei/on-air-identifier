#!/bin/bash
# v12 stage 1: whole-race dense pictures (code/v12_dense.py) on the development days, then on the six benchmark test days (seen-day check only).
cd "$(dirname "$0")/.."
PY=/home/ohead/ai-workspace/.venv/bin/python
for D in otherday otherday2 otherday3 otherday4 otherday5 20261115 otherday6 otherday7 otherday8 otherday9 otherday10 otherday11; do
  until flock -n ~/icip2027_research_gpu.lock true; do sleep 60; done
  $PY code/v12_dense.py $D > v12/dense_$D.log 2>&1
  echo "$D $(date -Is) $(tail -1 v12/dense_$D.log)" >> v12/STATUS
done
echo "DONE $(date -Is)" >> v12/STATUS
