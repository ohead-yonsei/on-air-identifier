#!/bin/bash
# caps3: the confirmation on two new days (2026-08-08, 07-26). One GPU job at a time; one scoring.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=caps3/test.log
DAYS="otherday18 otherday19"
echo "[$(date -Is)] test start" >> $L
if sha256sum -c caps3/DESIGN_LOCK.sha256 > caps3/lock_check.txt 2>&1 && sha256sum -c caps3/DAYS_LOCK.sha256 >> caps3/lock_check.txt 2>&1; then echo "[$(date -Is)] lock verified" >> $L; else echo "[$(date -Is)] LOCK MISMATCH, stopping" >> $L; exit 1; fi
bash finish3/run_download.sh $DAYS
echo "[$(date -Is)] videos: $(ls otherday18/videos/*.mp4 2>/dev/null | wc -l) $(ls otherday19/videos/*.mp4 2>/dev/null | wc -l)" >> $L
$PY code/g1_official_days.py /tmp/onair_kra_raw otherday18=2026-08-08 otherday19=2026-07-26 >> $L 2>&1
for D in $DAYS; do
  $PY code/fn2_times.py $D finish2/times/$D.json > finish2/times/$D.log 2>&1 &
done
wait
echo "[$(date -Is)] official results and clock times ready" >> $L
for D in $DAYS; do
  $PY code/g4_dense_whole.py $D > caps3/dense_$D.log 2>&1
  echo "[$(date -Is)] whole-race pictures $D exit $?: $(tail -1 caps3/dense_$D.log)" >> $L
  $PY code/g10_dense_last.py $D > caps3/dense_last_$D.log 2>&1
  echo "[$(date -Is)] last-37-s pictures $D exit $?: $(tail -1 caps3/dense_last_$D.log)" >> $L
done
$PY code/h4_reprob.py cnn/models/caps2_a.pt A $DAYS > caps3/reprob_test_A.log 2>&1
$PY code/h4_reprob.py cnn/models/caps2_b.pt B $DAYS > caps3/reprob_test_B.log 2>&1
echo "[$(date -Is)] both classifiers on the new days (exit $?)" >> $L
$PY code/h8_eval_combo.py caps3/eval_test.json B:B:old B+M:B:B -- $DAYS > caps3/eval_test.txt 2>&1
$PY code/h9_verdict.py caps3/eval_test.json >> caps3/eval_test.txt 2>&1
echo "[$(date -Is)] scored (exit $?)" >> $L
echo "[$(date -Is)] test done" >> $L
