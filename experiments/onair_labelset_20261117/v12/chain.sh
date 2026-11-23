#!/bin/bash
# v12 test (v12/DESIGN_KO.md): after the other GPU jobs, run live_run.py --finish-leader --v12 at real speed on 2026-10-25 and 10-18, then score.
# run on the server: setsid nohup bash v12/chain.sh &      progress in v12/TEST_STATUS
set -u
cd "$(dirname "$0")/.."
PY=/home/ohead/ai-workspace/.venv/bin/python
ST=v12/TEST_STATUS
echo "waiting $(date -Is)" > $ST
until grep -q "download done" v12/download.log 2>/dev/null; do sleep 30; done
until grep -q "^DONE" v12/STATUS 2>/dev/null && grep -q "^DONE" stt1/STATUS 2>/dev/null; do sleep 60; done     # no other GPU work of ours during the timed runs
until flock -n ~/icip2027_research_gpu.lock true; do sleep 60; done
echo "downloads: $(ls otherday28/videos/*.mp4 2>/dev/null | wc -l) + $(ls otherday29/videos/*.mp4 2>/dev/null | wc -l) videos; start $(date -Is)" >> $ST
mkdir -p v12/cards
for spec in "otherday28 20261025" "otherday29 20261018"; do
  set -- $spec
  for N in 1 2 3 4 5 6 7 8 9 10 11; do
    R=$(printf "R%02d" "$N"); O=v12/test/$2/$R
    V=$(ls $1/videos/${R}_*.mp4 2>/dev/null | head -1)
    mkdir -p "$O"
    if [ -z "$V" ]; then echo "$R no video" > "$O/run.log"; continue; fi
    [ -s v12/cards/$2/$R/racecard.json ] || $PY code/ls1_racecard.py "$2" "$N" v12/cards/$2/$R v12/raw > v12/cards/card_${2}_$R.log 2>&1
    [ -s v12/cards/$2/$R/racecard.json ] || { echo "$R no racecard" > "$O/run.log"; continue; }
    cp v12/cards/$2/$R/racecard.json v12/cards/$2/$R/official.json "$O/"
    $PY identifier/live_run.py --video "$V" --racecard "$O/racecard.json" --out-dir "$O" --no-video --finish-leader --v12 > "$O/run.log" 2>&1
    echo "$2 $R $(date -Is) $(tail -1 "$O/run.log" | cut -c1-120)" >> $ST
  done
done
$PY code/v12_score.py v12/test 20261025 20261018 > v12/test_score.log 2>&1
echo "DONE $(date -Is)" >> $ST
