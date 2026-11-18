#!/bin/bash
# finish3 stage 2: the two new days (2026-08-22, 08-16). Official results and clock times, then the pictures (one GPU job at a time), then one scoring.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=finish3/stage2.log
echo "[$(date -Is)] stage 2 start" >> $L
if sha256sum -c finish3/DESIGN2_LOCK.sha256 > finish3/lock2_check.txt 2>&1; then echo "[$(date -Is)] lock verified" >> $L; else echo "[$(date -Is)] LOCK MISMATCH, stopping" >> $L; exit 1; fi
$PY code/g1_official_days.py /tmp/onair_kra_raw otherday14=2026-08-22 otherday15=2026-08-16 >> $L 2>&1
for D in otherday14 otherday15; do
  $PY code/fn2_times.py $D finish2/times/$D.json > finish2/times/$D.log 2>&1 &
done
wait
echo "[$(date -Is)] official results and clock times ready" >> $L
for D in otherday14 otherday15; do
  $PY code/g4_dense_whole.py $D > finish3/dense_$D.log 2>&1
  echo "[$(date -Is)] whole-race pictures $D exit $?: $(tail -1 finish3/dense_$D.log)" >> $L
  $PY code/g10_dense_last.py $D > finish3/dense_last_$D.log 2>&1
  echo "[$(date -Is)] last-37-s pictures $D exit $?: $(tail -1 finish3/dense_last_$D.log)" >> $L
done
$PY code/g11_score2.py finish3/score_stage2.json otherday14 otherday15 > finish3/score_stage2.txt 2>&1
echo "[$(date -Is)] stage 2 scored (exit $?)" >> $L
echo "[$(date -Is)] stage 2 done" >> $L
