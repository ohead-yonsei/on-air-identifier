"""stt1: transcribe every downloaded commentary track once (local GPU, Whisper, English: the race videos carry English commentary).
usage (server): transcribe_all.py      reads stt1/audio/FOLDER/RNN_ID.m4a, writes stt1/out/FOLDER/RNN_ID.json; skips finished files.
No prompt: a Korean horse-name prompt made Whisper loop (pilot, 2026-11-23). The commentary is a reference only; the identifier never reads it."""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

S = Path(__file__).resolve().parent
MODEL = "openai/whisper-large-v3-turbo"
proc = AutoProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
files = sorted(S.glob("audio/*/*.m4a"))
for i, f in enumerate(files):
    out = S / "out" / f.parent.name / (f.stem + ".json")
    if out.exists():
        continue
    t0 = time.time()
    try:
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(f), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
        wav = np.frombuffer(pcm, dtype=np.float32)
        inp = proc(wav, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True)
        if inp["input_features"].shape[-1] < 3000:
            inp = proc(wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            res = model.generate(input_features=inp["input_features"].cuda().half(), attention_mask=inp["attention_mask"].cuda(),
                                 language="en", task="transcribe", return_timestamps=True, return_segments=True, condition_on_prev_tokens=False)
        segs = [{"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "text": proc.decode(s["tokens"], skip_special_tokens=True).strip()}
                for s in res["segments"][0]]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"audio": str(f.relative_to(S.parent)), "model": MODEL, "language": "en", "seconds": round(len(wav) / 16000, 1),
                                   "segments": segs}, ensure_ascii=False, indent=1))
        print(f"{i + 1}/{len(files)} {f.parent.name}/{f.stem} {len(segs)} segments {time.time() - t0:.1f} s", flush=True)
    except Exception as e:
        print(f"{i + 1}/{len(files)} {f.parent.name}/{f.stem} FAILED {e!r}", flush=True)
