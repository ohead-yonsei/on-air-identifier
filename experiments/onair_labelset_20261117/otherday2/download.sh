#!/bin/bash
# Download 2026-11-07 Seoul R1-R10 (KRBC International official uploads). Video only, <=1080p, H.264 preferred. Internal research use only.
set -u
O=/home/ohead/ai-workspace/projects/onair_labelset_20261117/otherday2
LIB=/home/ohead/ai-workspace/projects/onair_overnight_20261117/tools/ytdlp-lib
mkdir -p "$O/videos"
cd "$O"
LOG=download.log
echo "start $(date -Is)" >> "$LOG"
/home/ohead/ai-workspace/.venv/bin/python - <<PY > ids.txt
import json
for v in json.load(open("video_ids.json"))["videos"]:
    print(v["race"], v["id"])
PY
while read -r R ID; do
  ok=0
  for attempt in 1 2; do
    echo "[$(date -Is)] $R $ID attempt $attempt" >> "$LOG"
    if env PYTHONPATH="$LIB" /home/ohead/ai-workspace/.venv/bin/python -m yt_dlp --js-runtimes node --no-cookies --no-playlist \
        -f "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]" --write-info-json --no-write-comments --no-progress \
        -o "videos/${R}_${ID}.%(ext)s" "https://www.youtube.com/watch?v=${ID}" >> "$LOG" 2>&1; then
      ok=1; break
    fi
    sleep 10
  done
  echo "[$(date -Is)] $R $ID result ok=$ok" >> "$LOG"
  sleep 3
done < ids.txt
echo "end $(date -Is)" >> "$LOG"
grep -c "result ok=1" "$LOG"
ls -la videos | grep -c "\.mp4$"
du -sh videos | cut -f1
