#!/bin/bash
# caps2: program check of --cap-weights (after the scoring; the lock covered identify_l1.py before this option was added).
#   1. the default output must not change (2026-08-23 against night/confirm2)
#   2. the identifier class with the new classifier (v10, leader bar 0.7) on the two test days, scored by g3_score.py, to compare with the
#      calculation on the picture data (h5_eval_caps.py)
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=caps2/program.log
echo "[$(date -Is)] start" >> $L
mkdir -p caps2/default_check
$PY identifier/identify_l1.py --day-dir otherday13 --out caps2/default_check/otherday13_default.json > caps2/default_check/otherday13_default.log 2>&1
echo "[$(date -Is)] default after adding --cap-weights: $($PY code/id_same.py caps2/default_check/otherday13_default.json night/confirm2/otherday13_f2.json | tail -1)" >> $L
for D in otherday16 otherday17; do
  $PY code/g13_whole_leader_sure.py $D 0.7 cnn/models/caps2_a.pt CAPS2 > caps2/whole_$D.log 2>&1
  echo "[$(date -Is)] $D exit $?: $(tail -1 caps2/whole_$D.log)" >> $L
done
$PY code/g3_score.py caps2/score_program.json CAPS2 otherday16 otherday17 > caps2/score_program.txt 2>&1
echo "[$(date -Is)] done" >> $L
