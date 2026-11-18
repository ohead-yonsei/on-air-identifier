#!/bin/bash
# 2026-11-19: check of the program's --v10 option (identifier/identify_l1.py). No downloads. One GPU job at a time.
#   1. the script-demo race (2026-11-08 R11) with the demo's own time grid, against code/ls2_run.py's output
#   2. the three confirmation days of finish2 (2026-09-12, 09-05, 08-30; 30 races), whole race, against the official record
#      and the experiment's winner calls. These days were already scored in finish2: a check of the program, not a new test.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
O=finish2/program_v10
L=$O/run.log
mkdir -p $O/demo
echo "[$(date -Is)] start" >> $L
D=live_script/20261108_R11
$PY identifier/identify_l1.py --video $D/videos/R11_vkJ2az4nXkc.video.mp4 --out $O/demo/R11.json --no-strip-check --v10 --start 17.867 --end 96.25 > $O/demo/R11.log 2>&1
echo "[$(date -Is)] demo exit $?: $($PY code/fn13_program_v10.py demo $O/demo/R11.json $D 2>&1 | tail -1)" >> $L
for DAY in otherday9 otherday10 otherday11; do
  mkdir -p $O/$DAY
  for V in $DAY/videos/R??_*.mp4; do
    R=$(basename "$V" | cut -c1-3)
    [ -s $O/$DAY/$R.json ] || $PY identifier/identify_l1.py --video "$V" --out $O/$DAY/$R.json --no-strip-check --v10 > $O/$DAY/$R.log 2>&1
  done
  echo "[$(date -Is)] $DAY: $(ls $O/$DAY/R??.json 2>/dev/null | wc -l) races" >> $L
done
$PY code/fn13_program_v10.py days finish2/program_v10_C.json otherday9 otherday10 otherday11 > $O/score.txt 2>&1
echo "[$(date -Is)] done" >> $L
