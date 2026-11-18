#!/bin/bash
# 2026-11-18: server queue after the user asked for a draft benchmark. One GPU job at a time:
# 1 a first sample of the picture-only baselines (300 items each)   2 finish2 confirmation days + whole race
# 3 the rest of the picture-only baselines   4 the rest of L1 on the measurement days
cd ~/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
BM=$HOME/ai-workspace/projects/onair_benchmark_v0
mkdir -p finish2/whole/log
$PY $BM/baseline_vlm.py --name vlm_qwen9b_free --split test --prompt free --limit 300 > $BM/baselines/vlm_qwen9b_free.log 2>&1
$PY $BM/baseline_vlm.py --name vlm_qwen9b_careful --split test --prompt careful --limit 300 > $BM/baselines/vlm_qwen9b_careful.log 2>&1
echo done > $BM/baselines/VLM_300_DONE
rm -f finish2/whole/F2/otherday5.json
$PY code/fn10_whole.py otherday5 R02 > finish2/whole/log/smoke.log 2>&1 && $PY code/fn11_score_whole.py finish2/explore/smoke_whole.json otherday5 >> finish2/whole/log/smoke.log 2>&1 || { echo failed > finish2/whole/SMOKE_FAILED; }
if [ ! -f finish2/whole/SMOKE_FAILED ]; then
  sha256sum -c --quiet <(grep -v "^#" finish2/DESIGN2_LOCK.sha256) > finish2/lock2_check_before_C.txt 2>&1; echo "exit $?" >> finish2/lock2_check_before_C.txt
  for d in otherday9 otherday10 otherday11; do $PY code/fn10_whole.py $d > finish2/whole/log/$d.log 2>&1; $PY code/fn6_dense.py $d > finish2/dense/$d.log 2>&1; done
  echo done > finish2/whole/C_DONE
  for d in otherday6 otherday7 otherday8; do $PY code/fn10_whole.py $d > finish2/whole/log/$d.log 2>&1; done
  echo done > finish2/whole/M_DONE
fi
$PY $BM/baseline_vlm.py --name vlm_qwen9b_free --split test --prompt free >> $BM/baselines/vlm_qwen9b_free.log 2>&1
$PY $BM/baseline_vlm.py --name vlm_qwen9b_careful --split test --prompt careful >> $BM/baselines/vlm_qwen9b_careful.log 2>&1
echo done > $BM/baselines/VLM_DONE
for d in otherday6 otherday7 otherday8; do $PY code/fn3_run.py $d L1 >> finish2/runs/log/L1_$d.log 2>&1; done
echo done > finish2/runs/M_L1_DONE
