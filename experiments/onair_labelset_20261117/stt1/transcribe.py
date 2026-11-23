"""stt1: transcribe race commentary audio with Whisper (local GPU, no paid API), with the race card's horse names as a prompt.
usage (server): transcribe.py AUDIO RACECARD.json OUT.json [--model openai/whisper-large-v3-turbo]
Writes segments [{start, end, text}] in seconds of the video. The commentary is a reference only; the identifier never reads it."""
import argparse
import json
import subprocess

import numpy as np
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("audio"); ap.add_argument("card"); ap.add_argument("out")
ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
ap.add_argument("--no-prompt", action="store_true")
ap.add_argument("--lang", default="en", help="the race videos carry English commentary")
a = ap.parse_args()

pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", a.audio, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
wav = np.frombuffer(pcm, dtype=np.float32)
card = json.load(open(a.card))
prompt = "경마 중계. 출전마 " + ", ".join(f"{r['number']}번 {r['name']}" for r in card["runners"]) + "."

proc = AutoProcessor.from_pretrained(a.model)
model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
inp = proc(wav, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True)
if inp["input_features"].shape[-1] < 3000:                       # short clip: pad to 30 s
    inp = proc(wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
kw = dict(language=a.lang, task="transcribe", return_timestamps=True, return_segments=True, condition_on_prev_tokens=False)
if not a.no_prompt:
    kw["prompt_ids"] = torch.tensor(proc.get_prompt_ids(prompt)).cuda()
    kw["prompt_condition_type"] = "first-segment"
with torch.no_grad():
    res = model.generate(input_features=inp["input_features"].cuda().half(), attention_mask=inp["attention_mask"].cuda(), **kw)
segs = []
for s in res["segments"][0]:
    text = proc.decode(s["tokens"], skip_special_tokens=True).strip()
    segs.append({"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "text": text})
json.dump({"audio": a.audio, "model": a.model, "prompt": None if a.no_prompt else prompt, "language": a.lang, "seconds": round(len(wav) / 16000, 1), "segments": segs},
          open(a.out, "w"), ensure_ascii=False, indent=1)
print(len(segs), "segments", round(len(wav) / 16000, 1), "s")
for s in segs:
    print(f"{s['start']:7.2f} {s['end']:7.2f}  {s['text']}")
