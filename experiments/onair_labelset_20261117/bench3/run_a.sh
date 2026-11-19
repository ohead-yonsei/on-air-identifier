#!/bin/bash
# bench3: sentences written by a language model (Korean, English) on the test split of both tracks, once. One GPU job at a time.
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
B=/home/ohead/ai-workspace/projects/onair_benchmark_v0
PY=/home/ohead/ai-workspace/.venv/bin/python
cd "$P" || exit 1
L=$P/bench3/run_a.log
TEST="otherday6 otherday7 otherday8 otherday9 otherday10 otherday11"
echo "[$(date -Is)] start" >> $L
if sha256sum -c bench3/DESIGN_LOCK.sha256 > bench3/lock_check.txt 2>&1; then echo "[$(date -Is)] lock verified" >> $L; else echo "[$(date -Is)] LOCK MISMATCH, stopping" >> $L; exit 1; fi
cd $B
T="$(for D in $TEST; do echo $B/ident/CAPS2_$D.json; done)"
W="$(for D in $TEST; do echo $P/bench2/whole/v11/$D.json; done)"

# 1. the fixed sentence pattern in English (no model)
$PY baseline_ours_llm.py $T --name ours_v11_template_en --lang en --template-only >> $L 2>&1
$PY baseline_ours_llm.py $W --name ours_v11_template_en_official --lang en --template-only --official >> $L 2>&1

# 2. the writer, Korean then English, strip track then official track
for LANG_ in ko en; do
  $PY baseline_ours_llm.py $T --name ours_v11_llm_${LANG_} --lang $LANG_ > $P/bench3/llm_${LANG_}_strip.log 2>&1
  echo "[$(date -Is)] writer $LANG_ strip (exit $?): $(tail -1 $P/bench3/llm_${LANG_}_strip.log)" >> $L
  $PY baseline_ours_llm.py $W --name ours_v11_llm_${LANG_}_official --lang $LANG_ --official > $P/bench3/llm_${LANG_}_official.log 2>&1
  echo "[$(date -Is)] writer $LANG_ official (exit $?): $(tail -1 $P/bench3/llm_${LANG_}_official.log)" >> $L
done
curl -s http://127.0.0.1:11435/api/generate -d '{"model": "qwen3.5:9b", "keep_alive": 0}' > /dev/null

# 3. tables
$PY make_table_llm.py --no-readme > $P/bench3/table.txt 2>&1
echo "[$(date -Is)] table (exit $?)" >> $L
echo "[$(date -Is)] bench3 done" >> $L
