#!/bin/bash
# live2: run identifier/live_run.py unchanged (live speed, no subtitled video) on the 2026-07-12 Seoul races, then score (live2/DESIGN_KO.md).
# run from the experiment root on the server: bash live2/run.sh
set -u
cd "$(dirname "$0")/.."
PY=/home/ohead/ai-workspace/.venv/bin/python
OUT=live2/20260712
mkdir -p "$OUT"
for N in 1 2 3 4 6 7 8 9 10; do
  R=$(printf "R%02d" "$N")
  V=$(ls otherday23/videos/${R}_*.mp4 2>/dev/null | head -1)
  mkdir -p "$OUT/$R"
  if [ -z "$V" ]; then echo "$R no video" | tee "$OUT/$R/run.log"; continue; fi
  $PY code/ls1_racecard.py 20260712 "$N" "$OUT/$R" live2/raw > "$OUT/$R/card.log" 2>&1
  $PY identifier/live_run.py --video "$V" --racecard "$OUT/$R/racecard.json" --out-dir "$OUT/$R" --no-video > "$OUT/$R/run.log" 2>&1 \
    && $PY code/lv1_score.py "$OUT/$R" "$OUT/$R/official.json" > "$OUT/$R/score.log" 2>&1
  echo "$R done: $(tail -1 "$OUT/$R/run.log" | cut -c1-200)"
done
$PY code/lv2_pool.py "$OUT" > live2/pool.log 2>&1
echo ALL DONE
