"""Stateless Ollama diagnostic; literal prompts and image hashes logged before each call."""
import argparse,base64,hashlib,json,pathlib,re,time,urllib.request,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
SYSTEM=("You are a live horse-race commentator. Describe what is happening at the end of the supplied sequence. "
"The pictures, when present, are in chronological order. Racecard rows describe runner identities, not running order. "
"Use visible cap colours or readable saddlecloth numbers to identify runners; the jacket is not the cap. "
"Determine running direction from the sequence; screen left-to-right position or proximity to the camera alone is not a ranking rule. "
"Give only the leading runners whose identities and order you can determine, up to three; a partial order is allowed. "
"If no leader can be determined, use unclear. Do not force a complete ranking. "
"First line: ORDER: followed by comma-separated horse numbers (maximum three) or unclear. "
"Then give English commentary of at most two sentences and at most 40 words. "
"The race distance is the total course length, not the distance remaining. Refer to a runner by number or the supplied English name.")
B_RULE=(" Prefer useful, visually supported live information whenever available. If identification or relative position is uncertain, "
"do not guess: continue with a relevant verified background fact from the supplied facts or racecard. "
"Do not abandon clear live information just to recite background facts.")
CONDITIONS=['F','F-shuffle','F-no-video','B']
def sha(b):return hashlib.sha256(b).hexdigest()
def dumps(x):return json.dumps(x,ensure_ascii=False)
def build(sc,c,cfg):
 card=cfg['racecard'];mapping={h['no']:h for h in card}
 order=cfg['shuffle_order'] if c=='F-shuffle' else [h['no'] for h in card]
 lines=[f"No.{n} | {mapping[n]['name_en']} ({mapping[n]['name_ko']}) | cap: {mapping[n]['cap']} | jockey: {mapping[n]['jockey_en']}" for n in order]
 text='RACECARD (identity information only)\n'+'\n'.join(lines)+'\n\nVERIFIED BACKGROUND FACTS\n'+'\n'.join('- '+f for f in cfg['fixed_facts'])+'\n\nCall the race now.'
 refs=sc['frames'] if c!='F-no-video' else []
 ims=[]
 for f in refs:
  b=(ROOT/f['file']).read_bytes();assert sha(b)==f['sha256'];ims.append(base64.b64encode(b).decode())
 messages=[{'role':'system','content':SYSTEM+(B_RULE if c=='B' else '')},{'role':'user','content':text}]
 if ims:messages[-1]['images']=ims
 body={'model':cfg['model'],'messages':messages,'stream':False,'think':False,'keep_alive':'10m','options':cfg['options']}
 manifest={'scene':sc['scene'],'condition':c,'racecard_order':order,'messages':[{'role':m['role'],'content':m['content']} for m in messages],'images':refs,'image_count':len(ims),'options':cfg['options'],'think':False,'model':cfg['model']}
 return body,manifest

def validate(txt,card):
 lines=txt.strip().splitlines();first=lines[0] if lines else '';valid_ids={h['no'] for h in card}
 order=None
 m=re.fullmatch(r'ORDER:\s*(unclear|\d+(?:\s*,\s*\d+){0,2})',first)
 if m:order=[] if m[1]=='unclear' else [int(x.strip()) for x in m[1].split(',')]
 body=' '.join(lines[1:]).strip();words=re.findall(r"\b[\w]+(?:[-'][\w]+)*\b",body)
 # Number references such as No.9 are not sentence boundaries.
 normalized=re.sub(r'\bNo\.(?=\d)','No',body)
 sentences=[x for x in re.split(r'[.!?]+(?:\s+|$)',normalized) if x.strip()]
 issues=[]
 if not m:issues.append('invalid_ORDER_line')
 elif order and (len(set(order))!=len(order) or not set(order)<=valid_ids):issues.append('invalid_runner_numbers')
 if not body:issues.append('empty_commentary')
 if len(words)>40:issues.append('over_40_words')
 if len(sentences)>2:issues.append('over_2_sentences')
 if re.search('[가-힣]',body):issues.append('non_English_commentary')
 return {'order':order,'word_count':len(words),'sentence_count':len(sentences),'issues':issues}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-new',type=int,default=1);ap.add_argument('--scene');ap.add_argument('--condition',choices=CONDITIONS);ap.add_argument('--write-prompts',action='store_true');a=ap.parse_args()
 cfg=json.loads((ROOT/'config.json').read_text());specs=json.loads((ROOT/'scene_inputs.json').read_text())
 tags=json.load(urllib.request.urlopen(cfg['endpoint']+'/api/tags',timeout=30));model=next(m for m in tags['models'] if m['name']==cfg['model']);assert model['digest']==cfg['model_digest'],'Model changed'
 out=ROOT/'raw_responses.jsonl';prior=[json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
 done={(r['scene'],r['condition']) for r in prior}
 pdir=ROOT/'prompts';pdir.mkdir(exist_ok=True)
 tasks=[]
 for sc in specs:
  assert sc.get('input_review_passed') is True,'Frames must be visually reviewed before inference'
  for c in CONDITIONS:
   b,m=build(sc,c,cfg);(pdir/f"{sc['scene']}__{c}.json").write_text(json.dumps(m,ensure_ascii=False,indent=2))
   if (not a.scene or sc['scene']==a.scene) and (not a.condition or c==a.condition) and (sc['scene'],c) not in done:tasks.append((b,m))
 if a.write_prompts:return
 assert len(prior)+min(a.max_new,len(tasks))<=20,'20-call cap'
 for b,m in tasks[:a.max_new]:
  payload=dumps(b).encode();record={**m,'attempt':1,'model_digest':cfg['model_digest'],'request_bytes':len(payload),'request_sha256':sha(payload),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  # Persist intent before network I/O, including hash of exact encoded request.
  with (ROOT/'request_ledger.jsonl').open('a') as f:f.write(dumps(record)+'\n')
  started=time.monotonic()
  try:
   req=urllib.request.Request(cfg['endpoint']+'/api/chat',data=payload,headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=600) as response:
    raw=response.read().decode();record.update(http_status=response.status,raw_http_response=raw,response_json=json.loads(raw))
   record['text']=record['response_json'].get('message',{}).get('content','')
   record['format']=validate(record['text'],cfg['racecard']);record['failure']=not bool(record['text'])
  except Exception as e:record.update(failure=True,error=repr(e))
  record['elapsed_seconds']=round(time.monotonic()-started,3)
  with out.open('a') as f:f.write(dumps(record)+'\n');f.flush()
  print(dumps({k:record.get(k) for k in ['scene','condition','elapsed_seconds','failure','text','format','error']}),flush=True)
if __name__=='__main__':main()
