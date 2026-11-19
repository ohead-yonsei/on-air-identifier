#!/bin/bash
# bench2: the official track of the draft benchmark (items the strip does not cover). One GPU job at a time.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
B=/home/ohead/ai-workspace/projects/onair_benchmark_v0
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
L=bench2/run_a.log
TEST="otherday6 otherday7 otherday8 otherday9 otherday10 otherday11"
DEV="otherday otherday2 otherday3 otherday4 otherday5"
echo "[$(date -Is)] start" >> $L
if sha256sum -c bench2/DESIGN_LOCK.sha256 > bench2/lock_check.txt 2>&1; then echo "[$(date -Is)] lock verified" >> $L; else echo "[$(date -Is)] LOCK MISMATCH, stopping" >> $L; exit 1; fi

# 1. items, pictures
$PY $B/build_items_official.py --exp $P > bench2/build_items.log 2>&1
echo "[$(date -Is)] items: $(tail -2 bench2/build_items.log | tr '\n' ' ')" >> $L
$PY $B/make_inputs.py --exp $P --items items_official.jsonl > bench2/make_inputs.log 2>&1
echo "[$(date -Is)] pictures (exit $?): $(tail -1 bench2/make_inputs.log)" >> $L

# 2. check: the run script with the earlier classifier gives finish2/whole/F2 again
$PY code/bw1_whole.py otherday9 check_v9 > bench2/whole_check_v9.log 2>&1
$PY code/bw2_same.py bench2/whole/check_v9/otherday9.json finish2/whole/F2/otherday9.json > bench2/check_v9.txt 2>&1
echo "[$(date -Is)] check against finish2/whole/F2: $(tail -1 bench2/check_v9.txt)" >> $L

# 3. v11 on the whole race, test days
for D in $TEST; do
  $PY code/bw1_whole.py $D v11 cnn/models/caps2_a.pt > bench2/whole_v11_$D.log 2>&1
  echo "[$(date -Is)] v11 whole $D exit $?: $(tail -1 bench2/whole_v11_$D.log)" >> $L
done

# 4. sentences of the baselines that need no language model
cd $B
$PY baseline_trivial.py --official >> $P/$L 2>&1
$PY baseline_ours.py $(for D in $TEST otherday5; do echo $P/finish2/whole/F2/$D.json; done) --name ours_identifier_F2_official --official >> $P/$L 2>&1
$PY baseline_ours.py $(for D in $TEST; do echo $P/bench2/whole/v11/$D.json; done) --name ours_identifier_v11_official --official >> $P/$L 2>&1

# 5. the model that only sees the pictures, both prompts, once
$PY baseline_vlm.py --name vlm_qwen9b_free_official --prompt free --items items_official.jsonl > $P/bench2/vlm_free.log 2>&1
echo "[$(date -Is)] vlm free (exit $?): $(tail -1 $P/bench2/vlm_free.log)" >> $P/$L
$PY baseline_vlm.py --name vlm_qwen9b_careful_official --prompt careful --items items_official.jsonl > $P/bench2/vlm_careful.log 2>&1
echo "[$(date -Is)] vlm careful (exit $?): $(tail -1 $P/bench2/vlm_careful.log)" >> $P/$L
curl -s http://127.0.0.1:11435/api/generate -d '{"model": "qwen3.5:9b", "keep_alive": 0}' > /dev/null

# 6. tables
$PY make_table_official.py > $P/bench2/table.txt 2>&1
echo "[$(date -Is)] table (exit $?)" >> $P/$L
echo "[$(date -Is)] test done" >> $P/$L

# 7. dev days with v11, for the next step (prompt work for model-written commentary): whole race, and the strip items
mkdir -p bench2/ident
cd $P
for D in $DEV; do
  $PY code/bw1_whole.py $D v11 cnn/models/caps2_a.pt > bench2/whole_v11_$D.log 2>&1
  echo "[$(date -Is)] v11 whole $D exit $?: $(tail -1 bench2/whole_v11_$D.log)" >> $L
  $PY identifier/identify_l1.py --day-dir $D --out bench2/ident/CAPS2_$D.json > bench2/ident_$D.log 2>&1
  echo "[$(date -Is)] v11 strip items $D exit $?" >> $L
done
echo "[$(date -Is)] all done" >> $L
