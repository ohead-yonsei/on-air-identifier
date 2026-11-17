"""Post-hoc input diagnostic, separate from the frozen 36-call paired experiment.
Six user-reviewed samples, helmet only, one image, no racing-silks registry.
The model predicts a colour/pattern class; code deterministically maps class to number.
"""
import base64,datetime,hashlib,json,time,urllib.request
from pathlib import Path
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
CLASSES=['unknown','white','yellow','red','black','blue','green','brown_maroon','pink','purple','sky_blue','sky_blue_white_stripes']
SYSTEM='You classify the visible colour and pattern of one cropped jockey helmet. Do not identify race position or read a racecard.'
PROMPT='''This image is a tight crop of one jockey helmet. Classify the helmet surface, ignoring skin, jacket, horse and background around its edge. Choose one colour/pattern class: white, yellow, red, black, blue, green, brown_maroon, pink, purple, sky_blue, sky_blue_white_stripes, or unknown. A solid light cap and a cap with alternating blue and white stripes are different classes. Return unknown if you cannot distinguish the class. Return only JSON with color_class.
HELMET: [img]'''
SCHEMA={'type':'object','properties':{'color_class':{'type':'string','enum':CLASSES}},'required':['color_class'],'additionalProperties':False}
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    cfg=json.loads((ROOT/'config.json').read_text());samples=json.loads((ROOT/'samples.json').read_text())
    dest=ROOT/'direct_cap';dest.mkdir(exist_ok=True)
    out=dest/'raw.jsonl';prior=[json.loads(s) for s in out.read_text().splitlines()] if out.exists() else []
    done={r['sample'] for r in prior}
    tags=json.load(urllib.request.urlopen(cfg['endpoint']+'/api/tags'))
    assert next(m for m in tags['models'] if m['name']==cfg['model'])['digest']==cfg['model_digest']
    for s in samples[6:]:
        if s['id'] in done:continue
        assert len(done)<6
        cap=Image.open(ROOT/s['cap_file']).convert('RGB')
        image=Image.new('RGB',(224,224),(96,96,96));patch=ImageOps.contain(cap,(224,224),Image.Resampling.LANCZOS)
        image.paste(patch,((224-patch.width)//2,(224-patch.height)//2));ip=dest/f"{s['id']}.png";image.save(ip)
        b=ip.read_bytes();messages=[{'role':'system','content':SYSTEM},{'role':'user','content':PROMPT,'images':[base64.b64encode(b).decode()]}]
        body={'model':cfg['model'],'messages':messages,'stream':False,'think':False,'keep_alive':'10m','options':cfg['options'],'format':SCHEMA}
        payload=json.dumps(body,ensure_ascii=False).encode()
        rec={'sample':s['id'],'stage':'posthoc_direct_cap','model':cfg['model'],'model_digest':cfg['model_digest'],'options':cfg['options'],'messages':[{'role':m['role'],'content':m['content']} for m in messages],'schema':SCHEMA,'image':str(ip.relative_to(ROOT)),'image_sha256':sha(b),'request_sha256':sha(payload),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':1,'failure':False}
        with (dest/'request_ledger.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        t=time.perf_counter()
        try:
            req=urllib.request.Request(cfg['endpoint']+'/api/chat',data=payload,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=180) as response:raw=response.read().decode();rec['http_status']=response.status
            data=json.loads(raw);parsed=json.loads(data['message']['content']);label=parsed['color_class']
            assert label in CLASSES
            rec.update(raw_http_response=raw,response_json=data,parsed=parsed,number=CLASSES.index(label))
        except Exception as e:rec.update(failure=True,error=repr(e))
        rec['http_elapsed_ms']=(time.perf_counter()-t)*1000
        with out.open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
        done.add(s['id'])
        print(json.dumps({k:rec.get(k) for k in ['sample','parsed','number','http_elapsed_ms','failure']}),flush=True)
    (dest/'design_note.json').write_text(json.dumps({'posthoc':True,'trigger':'All 12 H-condition predictions were unknown despite visually identifiable helmets.','changes':['Remove the silks reference image','Show cap alone at larger relative size','Replace runner identification with explicit colour/pattern classification and deterministic number lookup'],'causal_limit':'Multiple input/prompt changes; cannot isolate any one factor. Does not replace or erase the original H condition.','max_calls':6,'training_steps':0},indent=2)+'\n')
if __name__=='__main__':main()
