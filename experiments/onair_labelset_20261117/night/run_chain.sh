#!/bin/bash
# Overnight chain 2026-11-18 (one GPU job at a time). Processing only; nothing is scored here. See night/DESIGN_KO.md.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=night/chain.log
echo "[$(date -Is)] chain start" >> $L
for D in otherday8 otherday9; do bash code/nt_day.sh $D; echo "[$(date -Is)] $D exit $?" >> $L; done
# 2026-11-15 again with the bundled program that now also gives the leader and the order (for the commentary demo with order)
O=$P/live_commentary/20261115
mkdir -p "$O/results_order"
for R in R01 R02 R03 R04 R05 R06 R07 R08 R09 R10 R11; do
  [ -f "$O/results_order/$R.json" ] || $PY identifier/identify_l1.py --video "$O/videos/$R.mp4" --out "$O/results_order/$R.json" > "$O/results_order/$R.log" 2>&1
done
echo "[$(date -Is)] 20261115 with order: $(ls $O/results_order/*.json | grep -vc pts) races" >> $L
for D in otherday10 otherday11; do bash code/nt_day.sh $D; echo "[$(date -Is)] $D exit $?" >> $L; done
echo "[$(date -Is)] chain done" >> $L
