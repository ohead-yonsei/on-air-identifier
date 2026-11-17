"""Independent, fixed-schema local VLM calls. Target labels never enter requests."""
import argparse, base64, datetime, hashlib, json, time, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SYSTEM='You identify a single horse-race runner from visible helmet and/or racing silks. This is identification, not running order. Do not infer identity from race position or registry order. Never guess when evidence is insufficient.'
PROMPT='''The first image is the verified racing-silks registry for the eleven entrants, labelled No.1 to No.11. It describes identity, not current running order.
SILKS REGISTRY: [img]

The second image contains two separately cropped parts of ONE target rider: helmet on the LEFT, racing jacket on the RIGHT. Grey areas are unavailable or masked. Either part may be absent. Grey masks remove helmet information from jacket crops. Ignore saddle, horse equipment, hands and white breeches; they are not jacket patterns. Match any available jacket colours and patterns to the registry; similar jackets may be ambiguous.
TARGET PARTS: [img]

Fixed helmet classes for this race: 1 white; 2 yellow; 3 red; 4 black; 5 blue; 6 green; 7 brown/maroon; 8 pink; 9 purple; 10 sky blue; 11 sky blue and white stripes. Distinguish a solid pale helmet from white/blue stripes. Use helmet class and/or the provided jacket registry, as visible. If evidence is insufficient or conflicting, return number 0 (unknown). Do not use information from other requests.
Return JSON with number (0 through 11), cap_class, and basis. cap_class describes the visible helmet only; use not_visible if that patch is absent, unreadable if unclear. basis is cap, silks, both, or unclear. No explanation.'''
SCHEMA={'type':'object','properties':{'number':{'type':'integer','minimum':0,'maximum':11},'cap_class':{'type':'string','enum':['white','yellow','red','black','blue','green','brown_maroon','pink','purple','sky_blue','sky_blue_white_stripes','unreadable','not_visible']},'basis':{'type':'string','enum':['cap','silks','both','unclear']}},'required':['number','cap_class','basis'],'additionalProperties':False}
def sha(b):return hashlib.sha256(b).hexdigest()
def dumps(x):return json.dumps(x,ensure_ascii=False)
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

def build(sample,condition,cfg):
    images=[];manifest=[]
    for rel in ['references/registry.png',sample['inputs'][condition]['file']]:
        b=(ROOT/rel).read_bytes()
        images.append(base64.b64encode(b).decode())
        manifest.append({'file':rel,'sha256':sha(b),'bytes':len(b)})
    assert manifest[1]['sha256']==sample['inputs'][condition]['sha256']
    msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':PROMPT,'images':images}]
    body={'model':cfg['model'],'messages':msgs,'stream':False,'think':False,'keep_alive':'10m','options':cfg['options'],'format':SCHEMA}
    meta={'sample':sample['id'],'condition':condition,'messages':[{'role':m['role'],'content':m['content']} for m in msgs],'images':manifest,'format_schema':SCHEMA,'model':cfg['model'],'model_digest':cfg['model_digest'],'options':cfg['options'],'think':False}
    return body,meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-new',type=int,default=1);ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text());samples=json.loads((ROOT/'samples.json').read_text())
    out=ROOT/'vlm_raw.jsonl';prior=rows(out);done={(r['sample'],r['condition']) for r in prior}
    (ROOT/'prompts').mkdir(exist_ok=True)
    tasks=[]
    # Each condition occupies each order position exactly four times.
    for i,sample in enumerate(samples):
        order=['H','S','HS'];order=order[i%3:]+order[:i%3]
        for c in order:
            t=time.perf_counter();body,meta=build(sample,c,cfg);prep=(time.perf_counter()-t)*1000
            (ROOT/'prompts'/f"{sample['id']}_{c}.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
            if (sample['id'],c) not in done:tasks.append((body,meta,prep))
    if a.prepare_only:return
    tags=json.load(urllib.request.urlopen(cfg['endpoint']+'/api/tags',timeout=20))
    assert next(m for m in tags['models'] if m['name']==cfg['model'])['digest']==cfg['model_digest']
    assert len(prior)+min(a.max_new,len(tasks))<=cfg['max_vlm_calls']
    for body,meta,prep in tasks[:a.max_new]:
        payload=dumps(body).encode();record={**meta,'input_load_base64_ms':prep,'request_sha256':sha(payload),'request_bytes':len(payload),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':1,'failure':False}
        with (ROOT/'vlm_request_ledger.jsonl').open('a') as f:f.write(dumps(record)+'\n')
        start=time.perf_counter()
        try:
            req=urllib.request.Request(cfg['endpoint']+'/api/chat',data=payload,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=180) as response:raw=response.read().decode();record['http_status']=response.status
            data=json.loads(raw);text=data.get('message',{}).get('content','');record.update(raw_http_response=raw,response_json=data,text=text)
            parsed=json.loads(text)
            valid=set(parsed)==set(SCHEMA['required']) and isinstance(parsed['number'],int) and not isinstance(parsed['number'],bool) and 0<=parsed['number']<=11 and parsed['cap_class'] in SCHEMA['properties']['cap_class']['enum'] and parsed['basis'] in SCHEMA['properties']['basis']['enum']
            record.update(parsed=parsed,format_valid=valid)
            if not valid:record.update(failure=True,error='schema violation')
        except Exception as e:record.update(failure=True,error=repr(e))
        record['http_elapsed_ms']=(time.perf_counter()-start)*1000
        with out.open('a') as f:f.write(dumps(record)+'\n');f.flush()
        print(dumps({k:record.get(k) for k in ['sample','condition','parsed','http_elapsed_ms','failure','error']}),flush=True)

if __name__=='__main__':main()
