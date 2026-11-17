#!/bin/bash
# Download 2026-09-06 Seoul R1-R10 broadcast videos (video only, <=1080p, H.264 preferred).
# Source: KRBC International official YouTube uploads. Internal research use only.
set -u
P=/home/ohead/ai-workspace/projects/onair_overnight_20261117
cd "$P"
Y=(env PYTHONPATH=tools/ytdlp-lib /home/ohead/ai-workspace/.venv/bin/python -m yt_dlp)
LOG=evidence/video_download.log
echo "start $(date -Is)" >> "$LOG"
while read -r R ID; do
  ok=0
  for attempt in 1 2; do
    echo "[$(date -Is)] $R $ID attempt $attempt" >> "$LOG"
    if "${Y[@]}" --js-runtimes node --no-cookies --no-playlist \
        -f "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]" \
        --write-info-json --no-write-comments --no-progress \
        -o "videos/${R}_${ID}.%(ext)s" \
        "https://www.youtube.com/watch?v=${ID}" >> "$LOG" 2>&1; then
      ok=1; break
    fi
    sleep 10
  done
  echo "[$(date -Is)] $R $ID result ok=$ok" >> "$LOG"
  sleep 3
done <<'EOF'
R01 IFg4OHM-wfA
R02 3ykqUmFIjNs
R03 y0y-rVJzvJw
R04 TbzSIX66FkY
R05 nvwAmro525M
R06 -3RgyWRsSmE
R07 MotP61eW-3M
R08 H0A9wR3JIOs
R09 wp1ZvMxaJd4
R10 SUW1c_aqQ7Y
EOF
echo "end $(date -Is)" >> "$LOG"
grep "result ok=" "$LOG" | tail -10
