#!/bin/bash
# finish3 stage 1, preparation (no GPU): official results and race clock times of the two days. Run after the lock.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
L=finish3/prep.log
echo "[$(date -Is)] prep start" >> $L
$PY code/g1_official_days.py /tmp/onair_kra_raw otherday12=2026-08-29 otherday13=2026-08-23 >> $L 2>&1
for D in otherday12 otherday13; do
  [ -s finish2/times/$D.json ] || $PY code/fn2_times.py $D finish2/times/$D.json > finish2/times/$D.log 2>&1 &
done
wait
echo "[$(date -Is)] prep done" >> $L
