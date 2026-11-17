#!/bin/bash
# One race day, start to finish: download (unless videos are already there), prepare frames and read the strip, usual detection + Qwen,
# adaptive detection, cap classifier. usage: nt_day.sh DAY_DIR      Needs DAY_DIR/video_ids.json. Internal research use only.
# Prints counts only; no scoring happens here. See night/DESIGN_KO.md.
set -u
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
LIB=/home/ohead/ai-workspace/projects/onair_overnight_20261117/tools/ytdlp-lib
D=$1
O=$P/$D
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
mkdir -p "$O/videos"
LOG=$O/pipeline.log
echo "[$(date -Is)] start $D" >> "$LOG"
$PY - "$O" <<'PY' > "$O/ids.txt"
import json, sys
for v in json.load(open(sys.argv[1] + "/video_ids.json"))["videos"]:
    print(v["race"], v["id"])
PY
while read -r R ID; do
  if [ -s "$O/videos/${R}_${ID}.mp4" ] && [ -s "$O/videos/${R}_${ID}.info.json" ]; then continue; fi
  for attempt in 1 2 3; do
    echo "[$(date -Is)] download $R $ID attempt $attempt" >> "$LOG"
    if env PYTHONPATH="$LIB" $PY -m yt_dlp --js-runtimes node --no-cookies --no-playlist \
        -f "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]" --write-info-json --no-write-comments --no-progress \
        -o "$O/videos/${R}_${ID}.%(ext)s" "https://www.youtube.com/watch?v=${ID}" >> "$LOG" 2>&1; then break; fi
    sleep 20
  done
done < "$O/ids.txt"
echo "[$(date -Is)] videos: $(ls "$O/videos"/*.mp4 2>/dev/null | wc -l)" >> "$LOG"
$PY code/nt_prepare.py "$D" >> "$LOG" 2>&1 || { echo "[$(date -Is)] FAILED prepare" >> "$LOG"; exit 2; }
$PY code/dx_identify.py "$D" >> "$LOG" 2>&1 || { echo "[$(date -Is)] FAILED identify" >> "$LOG"; exit 3; }
$PY code/da1_run.py "$D" "$D" >> "$LOG" 2>&1 || { echo "[$(date -Is)] FAILED adaptive detection" >> "$LOG"; exit 4; }
$PY code/wk2_apply.py all_two_days.pt "da/$D" cnn_probs_two >> "$LOG" 2>&1 || { echo "[$(date -Is)] FAILED classifier" >> "$LOG"; exit 5; }
echo "[$(date -Is)] done $D" >> "$LOG"
