#!/bin/bash
# Overnight chain, second part: waits for run_chain.sh to finish, then the program check, timing, commentary with order, and scoring.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=night/chain2.log
until grep -q "chain done" night/chain.log 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] chain2 start" >> $L
mkdir -p night/check night/timing
O=$P/live_commentary/20261115
# 1. the fast program on one new day, compared with the scoring scripts' F1
$PY identifier/identify_l1.py --day-dir otherday8 --fast --out night/check/otherday8_fast.json > night/check/otherday8_fast.log 2>&1
$PY code/nt4_check_fast.py otherday8 night/check/otherday8_fast.json > night/check/otherday8_fast_check.txt 2>&1
echo "[$(date -Is)] fast check: $(tail -1 night/check/otherday8_fast_check.txt)" >> $L
# 2. timing on 2026-11-15 R03, no strip check: F1 twice (first run warms the disk cache), then L1
for RUN in fast_a fast_b; do $PY identifier/identify_l1.py --video "$O/videos/R03.mp4" --fast --no-strip-check --start 23 --end 87 --out night/timing/R03_$RUN.json > night/timing/R03_$RUN.log 2>&1; done
$PY identifier/identify_l1.py --video "$O/videos/R03.mp4" --no-strip-check --start 23 --end 87 --out night/timing/R03_l1.json > night/timing/R03_l1.log 2>&1
echo "[$(date -Is)] timing done" >> $L
# 3. the fast program on every race of 2026-11-15 (same time points as the L1 run; the strip is joined from that run when scoring)
mkdir -p "$O/results_fast"
for R in R01 R02 R03 R04 R05 R06 R07 R08 R09 R10 R11; do
  S=$($PY -c "import json;d=json.load(open('$O/results_order/$R.json'));print(d['start'], d['end'])")
  set -- $S
  $PY identifier/identify_l1.py --video "$O/videos/$R.mp4" --fast --no-strip-check --start $1 --end $2 --out "$O/results_fast/$R.json" > "$O/results_fast/$R.log" 2>&1
done
echo "[$(date -Is)] 20261115 fast: $(ls $O/results_fast/R??.json | wc -l) races" >> $L
# 4. commentary with order, and its scoring
$PY code/cl5_commentary_order.py 20261115 > "$O/commentary_v3.log" 2>&1
python3 code/cl6_score_order.py 20261115 > "$O/score_v3.txt" 2>&1
echo "[$(date -Is)] commentary v3 done" >> $L
# 5. scoring of the four new days
$PY code/nt3_score.py night/score.json otherday8 otherday9 otherday10 otherday11 > night/score.txt 2>&1
echo "[$(date -Is)] chain2 done" >> $L
