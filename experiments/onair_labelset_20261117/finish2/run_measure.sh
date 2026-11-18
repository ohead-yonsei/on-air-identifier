#!/bin/bash
# stage 1: after the development runs, fill in the development races skipped by the first anchor rule, then the measurement days (F2),
# then the dense pictures of the development days (stage 2 data), then the measurement days with L1 (slow)
cd ~/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
while [ ! -f finish2/runs/DEV_F2_DONE ]; do sleep 20; done
sha256sum -c --quiet <(grep -v "^#" finish2/DESIGN_LOCK.sha256) > finish2/lock_check_before_M.txt 2>&1; echo "exit $?" >> finish2/lock_check_before_M.txt
for d in otherday4 otherday3 otherday2 otherday otherday5 20261115; do $PY code/fn3_run.py $d F2 >> finish2/runs/log/F2_$d.log 2>&1; done
for d in otherday6 otherday7 otherday8; do $PY code/fn3_run.py $d F2 > finish2/runs/log/F2_$d.log 2>&1; done
echo done > finish2/runs/M_F2_DONE
mkdir -p finish2/dense
for d in otherday4 otherday3 otherday2 otherday otherday5 20261115; do $PY code/fn6_dense.py $d > finish2/dense/$d.log 2>&1; done
echo done > finish2/dense/DEV_DONE
for d in otherday6 otherday7 otherday8; do $PY code/fn3_run.py $d L1 > finish2/runs/log/L1_$d.log 2>&1; done
echo done > finish2/runs/M_L1_DONE
