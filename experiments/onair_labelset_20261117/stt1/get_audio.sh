#!/bin/bash
# stt1: download the audio track (commentary) of race videos already listed in DAY_DIR/ids.txt. No cookies, nothing worked around.
# usage (server): stt1/get_audio.sh DAY_DIR [RACE ...]      writes stt1/audio/DAY_DIR/RNN_ID.m4a
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
LIB=/home/ohead/ai-workspace/projects/onair_overnight_20261117/tools/ytdlp-lib
cd "$P" || exit 1
D=$1; shift
O=stt1/audio/$D; mkdir -p "$O"
while read -r R ID; do
  [ $# -gt 0 ] && [[ ! " $* " =~ " $R " ]] && continue
  [ -s "$O/${R}_${ID}.m4a" ] && continue
  for attempt in 1 2 3; do
    if env PYTHONPATH="$LIB" $PY -m yt_dlp --js-runtimes node --no-cookies --no-playlist -f "ba[ext=m4a]/ba" --no-progress \
        -o "$O/${R}_${ID}.%(ext)s" "https://www.youtube.com/watch?v=${ID}" >> stt1/download_detail.log 2>&1; then break; fi
    sleep 20
  done
  echo "[$(date -Is)] $D $R $(ls $O/${R}_${ID}.* 2>/dev/null | head -1)" >> stt1/download.log
done < "$D/ids.txt"
