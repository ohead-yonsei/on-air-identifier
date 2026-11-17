import hashlib,json,pathlib,sys
import run_diagnostic as runner
R=pathlib.Path(__file__).resolve().parents[1]
cfg=json.loads((R/'config.json').read_text());scenes={s['scene']:s for s in json.loads((R/'scene_inputs.json').read_text())};rows=[json.loads(x) for x in (R/'raw_responses.jsonl').read_text().splitlines()]
checks=[]
def check(label,ok):
 assert ok,label;checks.append({'check':label,'passed':True})
check('Exactly 16 unique scene/condition calls',len(rows)==16 and len({(r['scene'],r['condition']) for r in rows})==16)
check('Single model digest / settings',all(r['model_digest']==cfg['model_digest'] and r['options']==cfg['options'] for r in rows))
check('No failures or format violations',all(not r['failure'] and not r['format']['issues'] for r in rows))
for s,sc in scenes.items():
 rr={r['condition']:r for r in rows if r['scene']==s}
 check(s+' image count 16/16/0/16',all(rr[c]['image_count']==(0 if c=='F-no-video' else 16) for c in runner.CONDITIONS))
 check(s+' F/no-video same literal text',rr['F']['messages']==rr['F-no-video']['messages'])
 check(s+' F/B same facts and card',rr['F']['messages'][1]==rr['B']['messages'][1])
 check(s+' shuffle changes rows only',sorted(rr['F']['messages'][1]['content'].splitlines())==sorted(rr['F-shuffle']['messages'][1]['content'].splitlines()) and rr['F']['messages'][0]==rr['F-shuffle']['messages'][0])
 check(s+' every image within window',all(sc['window_start']-1e-6<=f['actual_pts_seconds']<sc['window_end_exclusive'] for f in sc['frames']))
 check(s+' 16 distinct frames',len({f['decoded_frame_index'] for f in sc['frames']})==16)
 for c,r in rr.items():
  body,manifest=runner.build(sc,c,cfg)
  check(s+' '+c+' exact request reproducible',hashlib.sha256(runner.dumps(body).encode()).hexdigest()==r['request_sha256'])
  check(s+' '+c+' two independent messages',len(body['messages'])==2 and [m['role'] for m in body['messages']]==['system','user'])
  if c!='F-no-video':check(s+' '+c+' same input images',r['images']==rr['F']['images'])
check('Four no-video requests identical',len({r['request_sha256'] for r in rows if r['condition']=='F-no-video'})==1)
check('No stochastic repetitions / all 4 conditions complete',len(rows)==4*len(scenes))
(R/'verification.json').write_text(json.dumps({'passed':len(checks),'checks':checks},indent=2));print('PASS',len(checks))
