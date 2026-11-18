#!/bin/bash
# finish3, night of 2026-11-19: one GPU job at a time, no downloads here.
#   0. wait for the program check of finish2/program_v10 to end, then put the new identify_l1.py (v10 option, strip check opt-in) in place
#      and verify the stage-1 lock
#   1. stage 1: v10 on the whole race on the two test days (g2_whole_v10.py), scored once (g3_score.py)
#   2. check that the program's default output did not change with the strip-check edit (2026-08-23, against night/confirm2)
#   3. stage 2 data: riders of every picture over the whole race, the twelve days seen before (g4_dense_whole.py)
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=finish3/night.log
echo "[$(date -Is)] night start, waiting for the program check" >> $L
until grep -q "\] done" finish2/program_v10/run.log; do sleep 30; done
cp identifier/identify_l1.py finish3/identify_l1_before.py
mv identifier/identify_l1.py.new identifier/identify_l1.py
if sha256sum -c finish3/DESIGN_LOCK.sha256 > finish3/lock_check.txt 2>&1; then echo "[$(date -Is)] lock verified" >> $L; else echo "[$(date -Is)] LOCK MISMATCH, stopping" >> $L; exit 1; fi
until grep -q "prep done" finish3/prep.log; do sleep 30; done
for D in otherday12 otherday13; do
  $PY code/g2_whole_v10.py $D > finish3/whole_$D.log 2>&1
  echo "[$(date -Is)] stage 1 run $D exit $?: $(tail -1 finish3/whole_$D.log)" >> $L
done
$PY code/g3_score.py finish3/score_v10.json V10 otherday12 otherday13 > finish3/score_v10.txt 2>&1
echo "[$(date -Is)] stage 1 scored (exit $?)" >> $L
mkdir -p finish3/default_check
$PY identifier/identify_l1.py --day-dir otherday13 --out finish3/default_check/otherday13_default.json > finish3/default_check/otherday13_default.log 2>&1
echo "[$(date -Is)] default after the strip-check edit: $($PY code/id_same.py finish3/default_check/otherday13_default.json night/confirm2/otherday13_f2.json | tail -1)" >> $L
for D in otherday otherday2 otherday3 otherday4 otherday5 20261115 otherday6 otherday7 otherday8 otherday9 otherday10 otherday11; do
  $PY code/g4_dense_whole.py $D > finish3/dense_$D.log 2>&1
  echo "[$(date -Is)] dense $D exit $?: $(tail -1 finish3/dense_$D.log)" >> $L
done
echo "[$(date -Is)] night chain done" >> $L
