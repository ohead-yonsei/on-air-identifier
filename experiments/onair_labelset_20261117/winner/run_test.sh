#!/bin/bash
# winner: the confirmation on new days. One GPU job at a time; one scoring.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=winner/test.log
ALL="otherday20=2026-07-25 otherday21=2026-07-19 otherday22=2026-07-18 otherday23=2026-07-12"
echo "[$(date -Is)] start" >> $L
if sha256sum -c winner/DESIGN_LOCK.sha256 > winner/lock_check.txt 2>&1 && sha256sum -c winner/DAYS_LOCK.sha256 >> winner/lock_check.txt 2>&1; then echo "[$(date -Is)] lock verified" >> $L; else echo "[$(date -Is)] LOCK MISMATCH, stopping" >> $L; exit 1; fi

# check 1: the new picture script against the earlier way, on a day already seen
$PY code/w5_dense_last.py otherday19 > winner/dense_otherday19.log 2>&1
$PY code/w4_winner_guard.py same otherday19 > winner/check_same.txt 2>&1
echo "[$(date -Is)] check 1: $(tail -1 winner/check_same.txt)" >> $L

# the new days: videos, official results, clock times (complete days only)
bash winner/run_download.sh otherday20 otherday21 otherday22 otherday23
DAYS=""; PAIRS=""
for X in $ALL; do
  D=${X%%=*}
  N=$(ls $D/videos/*.mp4 2>/dev/null | wc -l)
  echo "[$(date -Is)] $D videos: $N" >> $L
  if [ "$N" = "10" ]; then DAYS="$DAYS $D"; PAIRS="$PAIRS $X"; fi
done
if [ $(echo $DAYS | wc -w) -lt 2 ]; then echo "[$(date -Is)] fewer than two complete days: no verdict" >> $L; echo "[$(date -Is)] test stopped" >> $L; exit 2; fi
$PY code/g1_official_days.py /tmp/onair_kra_raw $PAIRS >> $L 2>&1
for D in $DAYS; do
  $PY code/fn2_times.py $D finish2/times/$D.json > winner/times_$D.log 2>&1 &
done
wait
echo "[$(date -Is)] official results and clock times ready for:$DAYS" >> $L
for D in $DAYS; do
  $PY code/w5_dense_last.py $D > winner/dense_$D.log 2>&1
  echo "[$(date -Is)] pictures $D exit $?: $(tail -1 winner/dense_$D.log)" >> $L
done
$PY code/w4_winner_guard.py score winner/score_test.json $DAYS > winner/score_test.txt 2>&1
echo "[$(date -Is)] scored (exit $?)" >> $L
echo "[$(date -Is)] test done" >> $L

# for the benchmark (days already seen): pictures for the three test days that have none yet, then the calls of all eleven days
for D in otherday6 otherday7 otherday8; do
  $PY code/w5_dense_last.py $D > winner/dense_$D.log 2>&1
  echo "[$(date -Is)] pictures $D exit $?: $(tail -1 winner/dense_$D.log)" >> $L
done
$PY code/w4_winner_guard.py calls winner/calls_bench.json otherday otherday2 otherday3 otherday4 otherday5 otherday6 otherday7 otherday8 otherday9 otherday10 otherday11 > winner/calls_bench.txt 2>&1
echo "[$(date -Is)] benchmark calls (exit $?): $(tail -1 winner/calls_bench.txt)" >> $L
echo "[$(date -Is)] all done" >> $L
