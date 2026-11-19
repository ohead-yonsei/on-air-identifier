#!/bin/bash
# caps3 (2026-11-19 afternoon): a second round of mined cap labels with the new classifier, and the position models trained again on its numbers.
# Everything here is on days seen before (choosing); the new-day test comes after a lock. One GPU job at a time.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=caps3/round2.log
MINE="otherday otherday2 otherday3 otherday4 otherday5 20261115 otherday6 otherday7 otherday8"
DEV="otherday otherday2 otherday3 otherday4 otherday5 20261115"
HELD="otherday9 otherday10 otherday11 otherday14 otherday15 otherday16 otherday17"
echo "[$(date -Is)] start" >> $L
$PY code/h4_reprob.py cnn/models/caps2_a.pt A $MINE > caps3/reprob_A_mine.log 2>&1
echo "[$(date -Is)] classifier A on the mining days (exit $?)" >> $L
for d in 2 3 4 5 12 13; do $PY code/wk2_apply.py caps2_a.pt da/otherday$d cnn_probs_A > caps3/apply_A_$d.log 2>&1; done
echo "[$(date -Is)] classifier A on the strip frames" >> $L
$PY code/h2_mine_caps_official.py caps3/mine2_r2 --source A $MINE > caps3/mine2_r2.log 2>&1
$PY code/h1_mine_caps.py caps3/mine_r2 --probs cnn_probs_A otherday2 otherday3 otherday4 otherday5 otherday12 otherday13 > caps3/mine_r2.log 2>&1
echo "[$(date -Is)] round 2 mined: $(tail -3 caps3/mine2_r2.log | head -1 | cut -c1-160) | strip: $(tail -3 caps3/mine_r2.log | head -1 | cut -c1-120)" >> $L
$PY code/h3_train_caps.py caps2_b.pt caps2/mine2 caps2/mine caps3/mine2_r2 caps3/mine_r2 > caps3/train_b.log 2>&1
echo "[$(date -Is)] classifier B trained: $(head -1 caps3/train_b.log | cut -c1-220)" >> $L
$PY code/h4_reprob.py cnn/models/caps2_b.pt B $HELD $DEV > caps3/reprob_B.log 2>&1
echo "[$(date -Is)] classifier B on the held and development days (exit $?)" >> $L
for d in 2 3 4 5; do $PY code/wk2_apply.py caps2_b.pt da/otherday$d cnn_probs_B > caps3/apply_B_$d.log 2>&1; done
$PY code/h7_train_models.py A > caps3/models_A.log 2>&1
$PY code/h7_train_models.py B > caps3/models_B.log 2>&1
echo "[$(date -Is)] position models: $(tail -1 caps3/models_A.log | cut -c1-80) | $(tail -1 caps3/models_B.log | cut -c1-80)" >> $L
$PY code/h8_eval_combo.py caps3/eval_held.json B:B:old A+M:A:A B+M:B:B -- $HELD > caps3/eval_held.txt 2>&1
echo "[$(date -Is)] done" >> $L
