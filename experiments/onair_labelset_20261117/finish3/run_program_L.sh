#!/bin/bash
# finish3: program check of --leader-sure 0.4 (after the stage-2 scoring; the lock of stage 2 covered identify_l1.py before this option was added).
#   1. the default output must not change (2026-08-23 against night/confirm2)
#   2. the class with the leader bar 0.4 on the two stage-2 days, scored by g3_score.py, to compare with the picture-data calculation (g11_score2.py)
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=finish3/program_L.log
echo "[$(date -Is)] start" >> $L
$PY identifier/identify_l1.py --day-dir otherday13 --out finish3/default_check/otherday13_default2.json > finish3/default_check/otherday13_default2.log 2>&1
echo "[$(date -Is)] default after adding --leader-sure: $($PY code/id_same.py finish3/default_check/otherday13_default2.json night/confirm2/otherday13_f2.json | tail -1)" >> $L
for D in otherday14 otherday15; do
  $PY code/g13_whole_leader_sure.py $D 0.4 > finish3/whole_L_$D.log 2>&1
  echo "[$(date -Is)] $D exit $?: $(tail -1 finish3/whole_L_$D.log)" >> $L
done
$PY code/g3_score.py finish3/score_program_L.json V10L otherday14 otherday15 > finish3/score_program_L.txt 2>&1
echo "[$(date -Is)] done" >> $L
