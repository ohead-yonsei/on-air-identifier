#!/bin/bash
# live3: run the live runner without (A) and with (B) --finish-leader on 2026-11-22 and 11-21, then B on 2026-07-12 (reference), then pool.
# run on the server from anywhere: setsid nohup bash live3/chain.sh &   (live3/DESIGN_KO.md)
set -u
cd "$(dirname "$0")/.."
PY=/home/ohead/ai-workspace/.venv/bin/python
ST=live3/STATUS
echo "waiting for downloads $(date -Is)" > $ST
until grep -q "download done" live3/download.log 2>/dev/null; do sleep 30; done
echo "downloads: $(ls otherday24/videos/*.mp4 2>/dev/null | wc -l) + $(ls otherday25/videos/*.mp4 2>/dev/null | wc -l) videos" >> $ST

one() {   # one ARM DAYDIR DATE N EXTRA...
  local ARM=$1 DD=$2 DATE=$3 N=$4; shift 4
  local R=$(printf "R%02d" "$N") O=live3/$ARM/$DATE/$(printf "R%02d" "$N")
  local V=$(ls $DD/videos/${R}_*.mp4 2>/dev/null | head -1)
  mkdir -p "$O"
  if [ -z "$V" ]; then echo "$R no video" > "$O/run.log"; return; fi
  [ -s live3/cards/$DATE/$R/racecard.json ] || $PY code/ls1_racecard.py "$DATE" "$N" live3/cards/$DATE/$R live3/raw > live3/cards/card_${DATE}_$R.log 2>&1
  [ -s live3/cards/$DATE/$R/racecard.json ] || { echo "$R no racecard" > "$O/run.log"; return; }
  cp live3/cards/$DATE/$R/racecard.json live3/cards/$DATE/$R/official.json "$O/"
  $PY identifier/live_run.py --video "$V" --racecard "$O/racecard.json" --out-dir "$O" --no-video "$@" > "$O/run.log" 2>&1 \
    && $PY code/lv1_score.py "$O" "$O/official.json" > "$O/score.log" 2>&1
  echo "$ARM $DATE $R $(date -Is) $(tail -1 "$O/run.log" | cut -c1-120)" >> $ST
}

mkdir -p live3/cards
for spec in "otherday24 20261122" "otherday25 20261121"; do
  set -- $spec
  for N in 1 2 3 4 5 6 7 8 9 10 11; do
    one A "$1" "$2" "$N"
    one B "$1" "$2" "$N" --finish-leader
  done
done
$PY code/lv4_pool.py live3 20261122 20261121 > live3/pool.log 2>&1
echo "locked test pooled $(date -Is)" >> $ST

# reference on a seen day (not in the verdicts): B on 2026-07-12; A is live2's run
for N in 1 2 3 4 6 7 8 9 10; do one B otherday23 20260712 "$N" --finish-leader; done
mkdir -p live3/A && rm -rf live3/A/20260712 && cp -r live2/20260712 live3/A/20260712
$PY code/lv4_pool.py live3 20260712 > live3/pool_20260712_reference.log 2>&1
echo "DONE $(date -Is)" >> $ST
