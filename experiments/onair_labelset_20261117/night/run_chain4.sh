#!/bin/bash
# Morning of 2026-11-18: the program as configuration F2 (--fast --fallback --leader-tau 0.5). No downloads. One GPU job at a time.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=night/chain4.log
echo "[$(date -Is)] chain4 start" >> $L
for D in otherday8 otherday10; do
  $PY identifier/identify_l1.py --day-dir $D --fast --fallback --leader-tau 0.5 --out night/check/${D}_f2.json > night/check/${D}_f2.log 2>&1
  $PY code/nc4_check_f2.py $D night/check/${D}_f2.json > night/check/${D}_f2_check.txt 2>&1
  echo "[$(date -Is)] f2 check: $(tail -1 night/check/${D}_f2_check.txt)" >> $L
done
# the plain --fast program must still give what it gave before the options were added
$PY identifier/identify_l1.py --day-dir otherday8 --fast --out night/check/otherday8_fast_again.json > night/check/otherday8_fast_again.log 2>&1
$PY code/nt4_check_fast.py otherday8 night/check/otherday8_fast_again.json > night/check/otherday8_fast_again_check.txt 2>&1
echo "[$(date -Is)] fast again: $(tail -1 night/check/otherday8_fast_again_check.txt)" >> $L
# 2026-11-15 as F2, same start and end as results_order, no strip reading (no language model at all)
O=$P/live_commentary/20261115
mkdir -p "$O/results_f2"
for R in R01 R02 R03 R04 R05 R06 R07 R08 R09 R10 R11; do
  SE=$($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['start'], d['end'])" "$O/results_order/$R.json")
  set -- $SE
  [ -f "$O/results_f2/$R.json" ] || $PY identifier/identify_l1.py --video "$O/videos/$R.mp4" --out "$O/results_f2/$R.json" --no-strip-check --start $1 --end $2 --fast --fallback --leader-tau 0.5 > "$O/results_f2/$R.log" 2>&1
done
echo "[$(date -Is)] 20261115 f2: $(ls $O/results_f2/*.json | grep -vc pts) races" >> $L
echo "[$(date -Is)] chain4 done" >> $L
