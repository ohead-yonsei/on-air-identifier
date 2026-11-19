#!/bin/bash
# far, stage 1 (days already seen): train the cap classifier with the far-shot labels, compare with v11's on seven days it never trained on.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=far/run_a.log
HELD="otherday16 otherday17 otherday18 otherday19 otherday20 otherday21 otherday22"
echo "[$(date -Is)] start" >> $L
$PY code/h3_train_caps.py caps_far_c.pt far/mine caps2/mine2 caps2/mine > far/train_c.log 2>&1
echo "[$(date -Is)] trained (exit $?): $(head -1 far/train_c.log | cut -c1-220) | $(tail -1 far/train_c.log)" >> $L
$PY code/x2_reprob.py cnn/models/caps_far_c.pt C $HELD > far/reprob_c.log 2>&1
echo "[$(date -Is)] probabilities recomputed (exit $?)" >> $L
$PY code/x3_eval.py far/eval_held.json C $HELD > far/eval_held.txt 2>&1
echo "[$(date -Is)] evaluated (exit $?)" >> $L
echo "[$(date -Is)] stage 1 done" >> $L
