#!/bin/bash
# 2026-11-18 late evening: after v9 became the command-line default of identifier/identify_l1.py, check that
#   (1) the program with no configuration options gives what --fast --fallback --leader-tau 0.5 gave (2026-08-23, 118 time points)
#   (2) --v8 gives what the program gave by default before (2026-10-03, races R01-R03, against leader/check/otherday7_program.json)
# No downloads. One GPU job at a time.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=night/chain6.log
O=night/default_check
mkdir -p $O
echo "[$(date -Is)] chain6 start" >> $L
$PY identifier/identify_l1.py --day-dir otherday13 --out $O/otherday13_default.json > $O/otherday13_default.log 2>&1
echo "[$(date -Is)] default vs earlier F2 output: $($PY code/id_same.py $O/otherday13_default.json night/confirm2/otherday13_f2.json | tail -1)" >> $L
echo "[$(date -Is)] default vs scoring script: $($PY code/nc4_check_f2.py otherday13 $O/otherday13_default.json | tail -1)" >> $L
$PY identifier/identify_l1.py --day-dir otherday7 --races R01 R02 R03 --v8 --out $O/otherday7_v8.json > $O/otherday7_v8.log 2>&1
$PY code/id_same.py $O/otherday7_v8.json leader/check/otherday7_program.json > $O/otherday7_v8_same.txt 2>&1
echo "[$(date -Is)] --v8 vs earlier default output: $(tail -1 $O/otherday7_v8_same.txt)" >> $L
echo "[$(date -Is)] chain6 done" >> $L
