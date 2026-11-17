#!/bin/bash
# Second confirmation, 2026-11-18 morning (one GPU job at a time). Processing only; nothing is scored here. See night/confirm2/DESIGN_KO.md.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=night/chain3.log
echo "[$(date -Is)] chain3 start" >> $L
for D in otherday12 otherday13; do bash code/nt_day.sh $D; echo "[$(date -Is)] $D exit $?" >> $L; done
echo "[$(date -Is)] chain3 done" >> $L
