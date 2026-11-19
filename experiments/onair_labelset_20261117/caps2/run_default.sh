#!/bin/bash
# caps2: after v11 became the command-line default of identifier/identify_l1.py (2026-11-19), check that
#   (1) --v9 gives what the program gave by default before (2026-08-23, 118 time points; and against the scoring script of night/confirm2)
#   (2) no option gives what --cap-weights cnn/models/caps2_a.pt gave before the change (2026-09-12, 128 time points)
#   (3) --v8 gives what it gave before (2026-10-03, races R01-R03)
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=caps2/default.log
O=caps2/default_check
mkdir -p $O
echo "[$(date -Is)] start" >> $L
$PY identifier/identify_l1.py --day-dir otherday13 --v9 --out $O/otherday13_v9.json > $O/otherday13_v9.log 2>&1
echo "[$(date -Is)] --v9 vs the default before: $($PY code/id_same.py $O/otherday13_v9.json night/confirm2/otherday13_f2.json | tail -1)" >> $L
echo "[$(date -Is)] --v9 vs the scoring script: $($PY code/nc4_check_f2.py otherday13 $O/otherday13_v9.json | tail -1)" >> $L
$PY identifier/identify_l1.py --day-dir otherday9 --out $O/otherday9_default.json > $O/otherday9_default.log 2>&1
echo "[$(date -Is)] default vs --cap-weights before: $($PY code/id_same.py $O/otherday9_default.json caps2/bench_ident/CAPS2_otherday9.json | tail -1)" >> $L
$PY identifier/identify_l1.py --day-dir otherday7 --races R01 R02 R03 --v8 --out $O/otherday7_v8.json > $O/otherday7_v8.log 2>&1
echo "[$(date -Is)] --v8 vs before: $($PY code/id_same.py $O/otherday7_v8.json leader/check/otherday7_program.json | tail -1)" >> $L
echo "[$(date -Is)] done" >> $L
