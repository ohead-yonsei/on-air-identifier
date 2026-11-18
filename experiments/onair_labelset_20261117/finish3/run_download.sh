#!/bin/bash
# finish3 stage 2: download the two confirmation days (video only, as code/nt_day.sh does; no cookies, nothing worked around). Internal research use only.
# usage: run_download.sh DAY_DIR [DAY_DIR ...]      needs DAY_DIR/video_ids.json
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
LIB=/home/ohead/ai-workspace/projects/onair_overnight_20261117/tools/ytdlp-lib
cd "$P" || exit 1
L=finish3/download.log
for D in "$@"; do
  O=$P/$D
  mkdir -p "$O/videos"
  echo "[$(date -Is)] start $D" >> $L
  $PY - "$O" <<'PY' > "$O/ids.txt"
import json, sys
for v in json.load(open(sys.argv[1] + "/video_ids.json"))["videos"]:
    print(v["race"], v["id"])
PY
  while read -r R ID; do
    if [ -s "$O/videos/${R}_${ID}.mp4" ] && [ -s "$O/videos/${R}_${ID}.info.json" ]; then continue; fi
    for attempt in 1 2 3; do
      echo "[$(date -Is)] download $D $R $ID attempt $attempt" >> $L
      if env PYTHONPATH="$LIB" $PY -m yt_dlp --js-runtimes node --no-cookies --no-playlist \
          -f "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]" --write-info-json --no-write-comments --no-progress \
          -o "$O/videos/${R}_${ID}.%(ext)s" "https://www.youtube.com/watch?v=${ID}" >> "$O/download_detail.log" 2>&1; then break; fi
      sleep 20
    done
  done < "$O/ids.txt"
  echo "[$(date -Is)] $D videos: $(ls "$O/videos"/*.mp4 2>/dev/null | wc -l)" >> $L
done
echo "[$(date -Is)] download done" >> $L
