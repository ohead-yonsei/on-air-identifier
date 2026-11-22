#!/bin/bash
# live4: the live3 comparison (A without, B with --finish-leader) repeated on 2026-11-01 and 10-31 (live4/DESIGN_KO.md). Starts after live3 is DONE.
# run on the server from anywhere: setsid nohup bash live4/chain.sh &   (live4/DESIGN_KO.md)
set -u
cd "$(dirname "$0")/.."
PY=/home/ohead/ai-workspace/.venv/bin/python
ST=live4/STATUS
echo "waiting for live3 and downloads $(date -Is)" > $ST
until grep -q "^DONE" live3/STATUS 2>/dev/null && grep -q "download done" live4/download.log 2>/dev/null; do sleep 60; done
echo "downloads: $(ls otherday26/videos/*.mp4 2>/dev/null | wc -l) + $(ls otherday27/videos/*.mp4 2>/dev/null | wc -l) videos" >> $ST

one() {   # one ARM DAYDIR DATE N EXTRA...
  local ARM=$1 DD=$2 DATE=$3 N=$4; shift 4
  local R=$(printf "R%02d" "$N") O=live4/$ARM/$DATE/$(printf "R%02d" "$N")
  local V=$(ls $DD/videos/${R}_*.mp4 2>/dev/null | head -1)
  mkdir -p "$O"
  if [ -z "$V" ]; then echo "$R no video" > "$O/run.log"; return; fi
  [ -s live4/cards/$DATE/$R/racecard.json ] || $PY code/ls1_racecard.py "$DATE" "$N" live4/cards/$DATE/$R live4/raw > live4/cards/card_${DATE}_$R.log 2>&1
  [ -s live4/cards/$DATE/$R/racecard.json ] || { echo "$R no racecard" > "$O/run.log"; return; }
  cp live4/cards/$DATE/$R/racecard.json live4/cards/$DATE/$R/official.json "$O/"
  $PY identifier/live_run.py --video "$V" --racecard "$O/racecard.json" --out-dir "$O" --no-video "$@" > "$O/run.log" 2>&1 \
    && $PY code/lv1_score.py "$O" "$O/official.json" > "$O/score.log" 2>&1
  echo "$ARM $DATE $R $(date -Is) $(tail -1 "$O/run.log" | cut -c1-120)" >> $ST
}

mkdir -p live4/cards
for spec in "otherday26 20261101" "otherday27 20261031"; do
  set -- $spec
  for N in 1 2 3 4 5 6 7 8 9 10 11; do
    one A "$1" "$2" "$N"
    one B "$1" "$2" "$N" --finish-leader
  done
done
$PY code/lv4_pool.py live4 20261101 20261031 > live4/pool.log 2>&1
echo "locked test pooled $(date -Is)" >> $ST


# the four unused days together (live3 + live4), the same verdicts (live4/DESIGN_KO.md)
for ARM in A B; do mkdir -p live4/all/$ARM; for X in live3/$ARM/20261122 live3/$ARM/20261121 live4/$ARM/20261101 live4/$ARM/20261031; do ln -sfn "$(pwd)/$X" live4/all/$ARM/; done; done
$PY code/lv4_pool.py live4/all 20261122 20261121 20261101 20261031 > live4/pool_all.log 2>&1
echo "DONE $(date -Is)" >> $ST

# afterwards, no GPU: which races have the race clock on screen at all (exploration, live4/DESIGN_KO.md)
$PY code/lv5_clock_scan.py live4/clock_scan.json > live4/clock_scan.log 2>&1
echo "CLOCK SCAN DONE $(date -Is)" >> $ST
