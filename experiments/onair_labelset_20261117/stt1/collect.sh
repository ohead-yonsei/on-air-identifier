#!/bin/bash
# stt1: download the commentary audio of every race video already on the server, then transcribe all of it.
# usage (server): setsid nohup bash stt1/collect.sh > stt1/collect.log 2>&1 &      progress in stt1/STATUS
P=/home/ohead/ai-workspace/projects/onair_labelset_20261117
PY=/home/ohead/ai-workspace/.venv/bin/python
LIB=/home/ohead/ai-workspace/projects/onair_overnight_20261117/tools/ytdlp-lib
cd "$P" || exit 1
ST=stt1/STATUS
echo "start $(date -Is)" > $ST
for V in otherday*/videos/*.mp4 ../onair_overnight_20261117/videos/*.mp4; do
  case "$V" in ../onair_overnight*) F=onair_overnight_20261117 ;; *) F=${V%%/*} ;; esac
  B=$(basename "$V" .mp4); ID=${B#*_}
  O=stt1/audio/$F; mkdir -p "$O"
  [ -s "$O/$B.m4a" ] && continue
  for attempt in 1 2 3; do
    env PYTHONPATH="$LIB" $PY -m yt_dlp --js-runtimes node --no-cookies --no-playlist -f "ba[ext=m4a]/ba" --no-progress \
      -o "$O/$B.%(ext)s" "https://www.youtube.com/watch?v=${ID}" >> stt1/download_detail.log 2>&1 && break
    sleep 20
  done
  [ -s "$O/$B.m4a" ] || echo "no audio $F/$B" >> $ST
done
echo "audio done $(ls stt1/audio/*/*.m4a | wc -l) files $(date -Is)" >> $ST
until flock -n ~/icip2027_research_gpu.lock true; do sleep 60; done      # yield to the other project's GPU job if it holds its lock
$PY stt1/transcribe_all.py >> stt1/transcribe.log 2>&1
echo "transcripts $(ls stt1/out/*/*.json | wc -l) $(date -Is)" >> $ST
echo "DONE $(date -Is)" >> $ST
