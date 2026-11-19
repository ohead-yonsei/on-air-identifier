#!/bin/bash
# bench3 prompt work on the dev split (round given as $1)
R=$1
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
B=/home/ohead/ai-workspace/projects/onair_benchmark_v0
PY=/home/ohead/ai-workspace/.venv/bin/python
cd $B
T="$(for D in otherday otherday2 otherday3 otherday4 otherday5; do echo $P/bench2/ident/CAPS2_$D.json; done)"
W="$(for D in otherday otherday2 otherday3 otherday4 otherday5; do echo $P/bench2/whole/v11/$D.json; done)"
for LANG_ in ko en; do
  $PY baseline_ours_llm.py $T --name dev${R}_llm_${LANG_} --lang $LANG_ --split dev --limit 150
  $PY baseline_ours_llm.py $W --name dev${R}_llm_${LANG_}_official --lang $LANG_ --split dev --limit 150 --official
done
echo "dev round $R done"
