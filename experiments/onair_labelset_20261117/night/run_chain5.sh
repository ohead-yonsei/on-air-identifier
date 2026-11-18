#!/bin/bash
# 2026-11-18 evening: program check of the second confirmation (night/confirm2/DESIGN_KO.md, "프로그램 확인").
# identify_l1.py --fast --fallback --leader-tau 0.5 on the two confirmation days, compared with the scoring script's F2 by code/nc4_check_f2.py.
# No downloads. One GPU job at a time.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=night/chain5.log
echo "[$(date -Is)] chain5 start" >> $L
for D in otherday12 otherday13; do
  $PY identifier/identify_l1.py --day-dir $D --fast --fallback --leader-tau 0.5 --out night/confirm2/${D}_f2.json > night/confirm2/${D}_f2.log 2>&1
  $PY code/nc4_check_f2.py $D night/confirm2/${D}_f2.json > night/confirm2/${D}_f2_check.txt 2>&1
  echo "[$(date -Is)] f2 check $D: $(tail -1 night/confirm2/${D}_f2_check.txt)" >> $L
done
echo "[$(date -Is)] chain5 done" >> $L
