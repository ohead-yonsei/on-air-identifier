"""Verify downloaded 2026-09-06 race videos and write evidence/video_sources.json.

Checks: title contains the expected date and race number, channel, format, resolution,
fps, duration, size, SHA256. Uses only stdlib + ffprobe.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

P = "/home/ohead/ai-workspace/projects/onair_overnight_20261117"
EXPECTED = {
    "R01": "IFg4OHM-wfA", "R02": "3ykqUmFIjNs", "R03": "y0y-rVJzvJw",
    "R04": "TbzSIX66FkY", "R05": "nvwAmro525M", "R06": "-3RgyWRsSmE",
    "R07": "MotP61eW-3M", "R08": "H0A9wR3JIOs", "R09": "wp1ZvMxaJd4",
    "R10": "SUW1c_aqQ7Y",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = [s for s in d["streams"] if s.get("codec_type") == "video"]
    a = [s for s in d["streams"] if s.get("codec_type") == "audio"]
    vs = v[0] if v else {}
    num, den = (vs.get("avg_frame_rate", "0/1").split("/") + ["1"])[:2]
    return {
        "codec": vs.get("codec_name"),
        "width": vs.get("width"),
        "height": vs.get("height"),
        "avg_fps": round(float(num) / float(den), 3) if float(den) else None,
        "nb_frames": vs.get("nb_frames"),
        "duration_s": round(float(d["format"].get("duration", 0)), 3),
        "audio_streams": len(a),
    }


def main():
    rows, problems = [], []
    for race, vid in EXPECTED.items():
        media = [p for p in glob.glob(os.path.join(P, "videos", f"{race}_{vid}.*"))
                 if not p.endswith((".json", ".part", ".ytdl"))]
        info_path = os.path.join(P, "videos", f"{race}_{vid}.info.json")
        row = {"race": race, "youtube_id": vid,
               "url": f"https://www.youtube.com/watch?v={vid}"}
        if not media:
            row["status"] = "missing"
            problems.append(f"{race}: file missing")
            rows.append(row)
            continue
        path = media[0]
        info = json.load(open(info_path)) if os.path.exists(info_path) else {}
        title = info.get("title", "")
        m = re.search(r"2026\.09\.06\s+R(\d+)", title)
        title_race = f"R{int(m.group(1)):02d}" if m else None
        row.update({
            "file": os.path.relpath(path, P),
            "size_bytes": os.path.getsize(path),
            "sha256": sha256(path),
            "title": title,
            "channel": info.get("channel") or info.get("uploader"),
            "upload_date": info.get("upload_date"),
            "format_id": info.get("format_id"),
            "yt_dlp_version": (info.get("_version") or {}).get("version"),
            "ffprobe": probe(path),
            "title_race": title_race,
        })
        row["status"] = "ok" if title_race == race else "title_mismatch"
        if row["status"] != "ok":
            problems.append(f"{race}: title says {title_race!r}: {title}")
        rows.append(row)
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Internal research use only. 2026-09-06 Seoul R1-R10, KRBC International official uploads.",
        "download": "video only (no audio), <=1080p, H.264 preferred; no cookies/login",
        "videos": rows,
        "problems": problems,
    }
    with open(os.path.join(P, "evidence", "video_sources.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for r in rows:
        fp = r.get("ffprobe", {})
        print(r["race"], r["status"], r.get("format_id"), f'{fp.get("width")}x{fp.get("height")}',
              fp.get("avg_fps"), fp.get("duration_s"), f'{r.get("size_bytes", 0)/1e6:.1f}MB',
              (r.get("sha256") or "")[:12], "|", r.get("title", "")[:70])
    print("problems:", problems or "none")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
