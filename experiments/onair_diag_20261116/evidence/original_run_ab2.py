"""Five-arm test (2026-11-16). Same task in every arm: say who leads, who is close behind, and where the field is.
  A = video + racecard
  R = A + rule (name only when sure; otherwise say you cannot tell; no unseen numbers or situations)
  F = A + verified facts (no rule)
  B = A + verified facts + rule (when unsure, fill with a verified fact)
  C = B + CONFIRMED ON SCREEN list from the identifier (silkid v4): only listed horses may be named
Local only (Ollama). usage: python3 run_ab2.py scenes.json <clips_dir> responses.jsonl --ident ~/onair_work/silkid/out4 [--conds A,R,F,B,C] [--limit N]"""
import argparse, json, os, subprocess, tempfile, time, glob

FPS, WINDOW, WIDTH = 2, 8.0, 768
TASK = ("You are the live race caller for this horse race. You get the racecard and the pictures of the last 8 seconds, in order. "
        "In three or four sentences say: which horse leads, which horses are closest behind it, and where on the track the field is. "
        "Use the horses' names. Identify a horse by the colour of its jockey's cap (helmet), using the racecard; do not go by the jacket. "
        "The saddle cloth has the same colour as the cap and carries the horse's number: use the number too if you can clearly read it. "
        "The 'SEOUL' graphic at the top left shows the race number, not a horse's number. "
        "The horses run anticlockwise. Depending on the camera, the leader can be at the left or at the right of the picture: "
        "work out which way is forward from the direction the horses face and move across the pictures. "
        "In a head-on shot the horse nearest the camera is in front. "
        "FORMAT: your first line must be exactly 'ORDER: ' followed by the cap colours you can see in RUNNING ORDER, leader first, "
        "separated by commas (running order, not left to right). Then an empty line, then your call.")
RULE_R = (" RULE: name a horse or give its position only when you clearly see its cap colour and you are sure. If you are not sure, say plainly "
          "that you cannot tell which horse it is. Do not state any distance, number or situation that you cannot see in the pictures.")
FACTS = " You also get VERIFIED FACTS. They are guaranteed correct and you may use them."
RULE_B = (" RULE: name a horse or give its current position only when you clearly see its cap colour and you are sure. Whenever you are not sure, "
          "do NOT guess a name and do NOT speak vaguely: say a verified fact instead (where the field is, the conditions, or a horse and its jockey). Do not state any distance or number that is "
          "not in the verified facts. Do not describe a situation you cannot clearly see.")
RULE_C = (" You also get a CONFIRMED ON SCREEN list from an identification system: these horses are certainly in the pictures. The list does NOT "
          "give the running order. You may name ONLY horses on this list as being in the pictures now, and give a position for one only if you "
          "clearly see it. If the list is empty, do not name any horse as being anywhere now. For everything else follow the rule above.")

def frames(clip, end):
    d = tempfile.mkdtemp()
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(max(0.0, end - WINDOW)), "-i", clip,
                    "-vf", f"fps={FPS},scale={WIDTH}:-2", "-q:v", "3", f"{d}/f%03d.jpg"], check=True)
    return sorted(glob.glob(f"{d}/*.jpg"))

def confirmed(sc, ident):
    """numbers the identifier is sure about in the two analysed frames of this scene (last frame and 2.5 s earlier)"""
    f = os.path.join(os.path.expanduser(ident), f"R{sc['race']}", "visible.json")
    if not os.path.exists(f): return None
    v = json.load(open(f)); nos = set()
    for ab in "ab":
        for d in v.get(f"{sc['scene']}_{ab}.jpg", []):
            if d.get("sure") and d.get("state", "sure") == "sure": nos.add(d["no"])
    return sorted(nos)

def messages(sc, cond, imgs, conf):
    card = "\n".join(f"No.{h['no']} {h['name']} - {h['cap']} cap" for h in sc["racecard"])
    text = f"RACECARD\n{card}\n"; system = TASK
    if cond == "R": system += RULE_R
    if cond in ("F", "B", "C"):
        system += FACTS; text += "\nVERIFIED FACTS\n" + "\n".join("- " + f for f in sc["facts"]) + "\n"
    if cond in ("B", "C"): system += RULE_B
    if cond == "C":
        system += RULE_C; by = {h["no"]: h for h in sc["racecard"]}
        text += "\nCONFIRMED ON SCREEN (order unknown)\n" + ("\n".join(f"- {by[n]['name']} ({by[n]['cap']} cap)" for n in conf if n in by) if conf else "- none") + "\n"
    text += f"\nThe pictures are the last {WINDOW:.0f} seconds, {FPS} frames per second, in order. Call the race now."
    return [{"role": "system", "content": system}, {"role": "user", "content": text, "images": imgs}]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes"); ap.add_argument("clips"); ap.add_argument("out")
    ap.add_argument("--ident", default="~/onair_work/silkid/out4"); ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--conds", default="A,R,F,B,C"); ap.add_argument("--reps", type=int, default=1); ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    scenes = json.load(open(a.scenes))[: a.limit]; conds = a.conds.split(",")
    if "C" in conds:
        miss = [s["scene"] for s in scenes if confirmed(s, a.ident) is None]
        if miss: raise SystemExit(f"identifier results not found for {len(miss)} scenes under {a.ident} (run silkid v4 first, or drop C with --conds A,R,F,B)")
    if a.dry_run:
        for c in conds: print(f"===== {c}\n" + json.dumps(messages(scenes[0], c, ["<16 frames>"], confirmed(scenes[0], a.ident) or []), indent=1, ensure_ascii=False))
        raise SystemExit
    import ollama
    done = {(r["scene"], r["cond"], r["rep"]) for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    t0 = time.time(); k = 0
    with open(a.out, "a") as fh:
        for n, sc in enumerate(scenes, 1):
            todo = [(c, r) for c in conds for r in range(a.reps) if (sc["scene"], c, r) not in done]
            if not todo: continue
            hit = glob.glob(os.path.join(os.path.expanduser(a.clips), sc["clip_glob"]))
            if not hit: print("NO CLIP", sc["scene"]); continue
            imgs = frames(hit[0], sc["video_end"]); conf = confirmed(sc, a.ident) if "C" in conds else None
            for c, r in todo:
                res = ollama.chat(model=a.model, messages=messages(sc, c, imgs, conf), options={"temperature": 0.2, "num_predict": 500, "num_ctx": 32768})
                txt = res["message"]["content"].strip(); k += 1
                fh.write(json.dumps({"scene": sc["scene"], "cond": c, "rep": r, "model": a.model, "response": txt, "confirmed": conf if c == "C" else None,
                                     "n_frames": len(imgs), "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False) + "\n"); fh.flush()
                print(f"[{n}/{len(scenes)}] {sc['scene']} {c}: {txt[:100]!r}")
            print(f"   elapsed {time.time()-t0:.0f}s, {k} calls, {(time.time()-t0)/max(1,k):.0f}s per call")
