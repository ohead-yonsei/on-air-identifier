#!/bin/bash
# exploration runs on the development days (finish stretch already looked at, or strip frames used for training / choosing)
cd ~/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
while [ ! -f finish2/times/DONE ]; do sleep 20; done
mkdir -p finish2/runs/log
for d in otherday4 otherday3 otherday2 otherday otherday5 20261115; do
  $PY code/fn3_run.py $d F2 > finish2/runs/log/F2_$d.log 2>&1
done
echo done > finish2/runs/DEV_F2_DONE
