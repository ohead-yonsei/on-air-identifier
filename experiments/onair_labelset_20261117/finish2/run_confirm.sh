#!/bin/bash
# stage 2 and 3: after L1 on the measurement days. Smoke test on one development race, then the confirmation days, then the measurement days.
cd ~/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
while [ ! -f finish2/runs/M_L1_DONE ]; do sleep 20; done
mkdir -p finish2/whole/log
$PY code/fn10_whole.py otherday5 R02 > finish2/whole/log/smoke.log 2>&1 && $PY code/fn11_score_whole.py finish2/explore/smoke_whole.json otherday5 >> finish2/whole/log/smoke.log 2>&1 || { echo failed > finish2/whole/SMOKE_FAILED; exit 1; }
sha256sum -c --quiet <(grep -v "^#" finish2/DESIGN2_LOCK.sha256) > finish2/lock2_check_before_C.txt 2>&1; echo "exit $?" >> finish2/lock2_check_before_C.txt
for d in otherday9 otherday10 otherday11; do $PY code/fn10_whole.py $d > finish2/whole/log/$d.log 2>&1; $PY code/fn6_dense.py $d > finish2/dense/$d.log 2>&1; done
echo done > finish2/whole/C_DONE
for d in otherday6 otherday7 otherday8; do $PY code/fn10_whole.py $d > finish2/whole/log/$d.log 2>&1; done
echo done > finish2/whole/M_DONE
