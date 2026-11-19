#!/bin/bash
# caps2 (2026-11-19): train the cap classifier with the mined labels, then compare it with the current one on five days that gave no labels
# (2026-09-12, 09-05, 08-30: the confirmation days of finish2; 08-22, 08-16: the stage-2 days of finish3). All of them were seen before:
# this is for choosing, not a test. One GPU job at a time.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=caps2/run_a.log
echo "[$(date -Is)] start" >> $L
$PY code/h3_train_caps.py caps2_a.pt caps2/mine2 caps2/mine > caps2/train_a.log 2>&1
echo "[$(date -Is)] trained: $(head -1 caps2/train_a.log | cut -c1-200) | $(tail -1 caps2/train_a.log)" >> $L
$PY code/h4_reprob.py cnn/models/caps2_a.pt A otherday9 otherday10 otherday11 otherday14 otherday15 > caps2/reprob_a.log 2>&1
echo "[$(date -Is)] probabilities recomputed (exit $?)" >> $L
$PY code/h5_eval_caps.py caps2/eval_a_held.json A otherday9 otherday10 otherday11 otherday14 otherday15 > caps2/eval_a_held.txt 2>&1
echo "[$(date -Is)] done" >> $L
