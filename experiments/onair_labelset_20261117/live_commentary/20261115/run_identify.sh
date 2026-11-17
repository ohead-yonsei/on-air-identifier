#!/bin/bash
# AI commentary demo, step 1: download 2026-11-15 Seoul R1-R11 (KRBC International official uploads; video only; internal research use)
# and run the bundled identifier (identifier/identify_l1.py) on every race. R01 is reused from demo/20261115_R01.
set -u
ROOT=/home/ohead/ai-workspace/projects/onair_labelset_20261117
O=$ROOT/live_commentary/20261115
LIB=/home/ohead/ai-workspace/projects/onair_overnight_20261117/tools/ytdlp-lib
PY=/home/ohead/ai-workspace/.venv/bin/python
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
mkdir -p "$O/videos" "$O/results" "$O/overlays"
cd "$O"
cp -n "$ROOT/demo/20261115_R01/R01_hCscAHwkZkY.mp4" videos/R01.mp4 2>/dev/null
while read -r R ID; do
  if [ ! -f "videos/$R.mp4" ]; then
    env PYTHONPATH="$LIB" $PY -m yt_dlp --js-runtimes node --no-cookies --no-playlist -f "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]" --no-progress -o "videos/$R.%(ext)s" "https://www.youtube.com/watch?v=$ID" >> download.log 2>&1
    sleep 3
  fi
done <<LIST
R02 cjxTS7R03UA
R03 zdt85sYjegA
R04 tBDDoYsKTdY
R05 SlMCqk9OgGk
R06 ZfdPCsNRs3Y
R07 Y_a-69puzOc
R08 qzuTmsa01fI
R09 -5KU39fyUi0
R10 y5Uxat-kvK8
R11 m1W7UZDVcdA
LIST
echo "videos: $(ls videos/*.mp4 | wc -l), $(du -sh videos | cut -f1)"
cd "$ROOT"
for R in R01 R02 R03 R04 R05 R06 R07 R08 R09 R10 R11; do
  if [ ! -f "$O/results/$R.json" ]; then
    $PY identifier/identify_l1.py --video "$O/videos/$R.mp4" --out "$O/results/$R.json" > "$O/results/$R.log" 2>&1
  fi
  $PY code/demo1_overlay.py "$O/videos/$R.mp4" "$O/results/$R.json" "$O/overlays/$R" 2>&1 | tail -1 | sed "s/^/$R: /"
done
date +%H:%M
