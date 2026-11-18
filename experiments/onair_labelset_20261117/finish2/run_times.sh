#!/bin/bash
cd ~/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
run() { $PY code/fn2_times.py "$1" "finish2/times/$2.json" > "finish2/times/$2.log" 2>&1; }
mkdir -p finish2/times
( run otherday otherday; run otherday5 otherday5; run otherday9 otherday9 ) &
( run otherday2 otherday2; run otherday6 otherday6; run otherday10 otherday10 ) &
( run otherday3 otherday3; run otherday7 otherday7; run otherday11 otherday11 ) &
( run otherday8 otherday8; run live_commentary/20261115 20261115 ) &
wait
echo done > finish2/times/DONE
