"""live_script: burn the stored commentary script into the race video as subtitles (a replay, not a live run).
usage: ls6_subtitle_video.py RACE_DIR VIDEO OUT.mp4 [IDENTIFIER_SECONDS]

Each sentence of RACE_DIR/script.json appears at the moment it would have been ready: its time point + the identifier's time per
time point (default 2.3 s, the measured value of the fast flow) + the time the writing model took for that sentence (all tries).
It stays until the next sentence is ready (6 s at most). The video is written without sound, 1280 px wide. Needs ffmpeg with libass."""
import json
import subprocess
import sys
from pathlib import Path

race, video, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
ident_s = float(sys.argv[4]) if len(sys.argv) > 4 else 2.3
rows = json.loads((race / "script.json").read_text())["rows"]
ready = [r["t"] + ident_s + sum(t.get("seconds", 0) for t in r.get("tries", [])) for r in rows]


def stamp(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


srt = []
for k, (r, a) in enumerate(zip(rows, ready)):
    b = min(ready[k + 1] if k + 1 < len(rows) else a + 6, a + 6)
    srt.append(f"{k + 1}\n{stamp(a)} --> {stamp(b - 0.05)}\n{r['line']}\n")
srt_path = out.with_suffix(".srt")
srt_path.write_text("\n".join(srt))
delay = [a - r["t"] for r, a in zip(rows, ready)]
note = {"sentences": len(rows), "identifier_seconds": ident_s, "delay_seconds_median": round(sorted(delay)[len(delay) // 2], 2),
        "delay_seconds_max": round(max(delay), 2), "written_by": {s: sum(r["source"] == s for r in rows) for s in {r["source"] for r in rows}}}
out.with_suffix(".json").write_text(json.dumps(note, ensure_ascii=False, indent=1))
label = f"화면만 보고 만든 자막 · 식별기 v9 + 로컬 모델 · 문장마다 약 {note['delay_seconds_median']:.1f}초 뒤에 표시 (다시 재생한 것)"
style = "FontName=Noto Sans CJK KR,FontSize=22,PrimaryColour=&H00FFFFFF,BackColour=&H90000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=28"
vf = (f"scale=1280:-2,subtitles={srt_path}:force_style='{style}',"
      f"drawtext=fontfile=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc:text='{label}':x=16:y=14:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf", vf, "-an", "-r", "30", "-c:v", "libx264", "-crf", "24", "-preset", "fast", str(out)], check=True)
print("written", out, note)
