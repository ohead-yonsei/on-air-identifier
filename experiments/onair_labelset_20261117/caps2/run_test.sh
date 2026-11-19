#!/bin/bash
# caps2: the confirmation on two new days (2026-08-15, 08-09). Download, official results and clock times, the pictures with the current
# classifier, the new classifier's probabilities on the same riders, one scoring. One GPU job at a time.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=caps2/test.log
DAYS="otherday16 otherday17"
echo "[$(date -Is)] test start" >> $L
if sha256sum -c caps2/DESIGN_LOCK.sha256 > caps2/lock_check.txt 2>&1 && sha256sum -c caps2/DAYS_LOCK.sha256 >> caps2/lock_check.txt 2>&1; then echo "[$(date -Is)] lock verified" >> $L; else echo "[$(date -Is)] LOCK MISMATCH, stopping" >> $L; exit 1; fi
bash finish3/run_download.sh $DAYS
echo "[$(date -Is)] videos: $(ls otherday16/videos/*.mp4 2>/dev/null | wc -l) $(ls otherday17/videos/*.mp4 2>/dev/null | wc -l)" >> $L
$PY code/g1_official_days.py /tmp/onair_kra_raw otherday16=2026-08-15 otherday17=2026-08-09 >> $L 2>&1
for D in $DAYS; do
  $PY code/fn2_times.py $D finish2/times/$D.json > finish2/times/$D.log 2>&1 &
done
wait
echo "[$(date -Is)] official results and clock times ready" >> $L
for D in $DAYS; do
  $PY code/g4_dense_whole.py $D > caps2/dense_$D.log 2>&1
  echo "[$(date -Is)] whole-race pictures $D exit $?: $(tail -1 caps2/dense_$D.log)" >> $L
  $PY code/g10_dense_last.py $D > caps2/dense_last_$D.log 2>&1
  echo "[$(date -Is)] last-37-s pictures $D exit $?: $(tail -1 caps2/dense_last_$D.log)" >> $L
done
$PY code/h4_reprob.py cnn/models/caps2_a.pt A $DAYS > caps2/reprob_test.log 2>&1
echo "[$(date -Is)] new classifier's probabilities (exit $?)" >> $L
$PY code/h5_eval_caps.py caps2/eval_a_test.json A $DAYS > caps2/eval_a_test.txt 2>&1
echo "[$(date -Is)] scored (exit $?)" >> $L
echo "[$(date -Is)] test done" >> $L
