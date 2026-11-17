"""Independent, resumable local VLM calls; full text/image hashes/raw responses."""
import argparse,base64,json,time,urllib.request
from PIL import Image,ImageOps
from common import ROOT,CFG,sha,dump,append,status,check_deadline,now

COLORS=['unknown','white','yellow','red','black','blue','green','brown_maroon','pink','purple','sky_blue','sky_blue_white_stripes']
VIEW_SCHEMA={'type':'object','properties':{'view':{'type':'string','enum':['side','front','rear','oblique_front','oblique_rear','unclear','not_racing']},'direction':{'type':'string','enum':['left','down_left','down','down_right','right','up_right','up','up_left','unclear']},'multiple_views':{'type':'boolean'},'evidence':{'type':'string'}},'required':['view','direction','multiple_views','evidence'],'additionalProperties':False}
LOC_SCHEMA={'type':'object','properties':{'leader_visible':{'type':'boolean'},'nose_xy':{'type':'array','items':{'type':'integer','minimum':0,'maximum':1000},'maxItems':2},'cap_box':{'type':'array','items':{'type':'integer','minimum':0,'maximum':1000},'maxItems':4},'cap_color':{'type':'string','enum':COLORS},'evidence':{'type':'string'}},'required':['leader_visible','nose_xy','cap_box','cap_color','evidence'],'additionalProperties':False}
SYSTEM='You inspect only the supplied horse-racing image. Do not use race results, popularity, assumed runner numbers, or any prior conversation. You may abstain when the visual evidence is insufficient.'
VIEW='Describe the camera view relative to the horses and the apparent direction they are progressing IN THIS IMAGE. Direction means the projected race progression, not camera panning or the direction a jockey looks. Choose unclear if not visually supported. Do not identify a leader or read any text. Evidence: at most 25 English words. IMAGE: [img]'
LOC='Identify the apparent leading horse only if visible and visually distinguishable. Do not force a full ranking. Locate the tip of that horse\'s muzzle and the helmet of the jockey RIDING THAT SAME HORSE. A nearby helmet may belong to another horse. Coordinates use x-right,y-down normalized 0..1000 over the entire image. nose_xy=[x,y]; cap_box=[left,top,right,bottom], tightly enclosing the helmet. If a landmark is not identifiable, return [] for it; if leader cannot be established, leader_visible=false. Classify only the helmet colour/pattern from the listed classes, otherwise unknown. Evidence: at most25 English words. IMAGE: [img]'

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['view','localize','conditioned'],required=True);p.add_argument('--first',action='store_true');p.add_argument('--mirror',action='store_true');p.add_argument('--max-new',type=int,default=100);p.add_argument('--model',default=CFG['model']);p.add_argument('--digest',default=CFG['model_digest']);p.add_argument('--core-only',action='store_true');p.add_argument('--front-first',action='store_true');a=p.parse_args()
 with urllib.request.urlopen(CFG['endpoint']+'/api/tags') as rr:tags=json.load(rr)
 assert next(m for m in tags['models'] if m['name']==a.model)['digest']==a.digest
 dump(ROOT/'evidence/installed_model_tags.json',tags)
 frames=json.loads((ROOT/'selection.json').read_text())['frames']
 if a.first:frames=[next(s for s in frames if s['target_seconds']==35)]
 if a.core_only and not a.first:frames=[s for s in frames if s['geometry_review']]
 if a.task in ['localize','conditioned']:frames=[s for s in frames if s['geometry_review'] or s['target_seconds']==35]
 if a.mirror:frames=[s for s in frames if s['geometry_review']]
 ann=json.loads((ROOT/'geometry_annotations.json').read_text()) if a.task=='conditioned' else None
 ann_lookup={f['id']:f for f in ann['frames']} if ann else {}
 out=ROOT/'raw/vlm.jsonl';ledger=ROOT/'raw/request_ledger.jsonl'
 prior=[json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
 done={r['item_id'] for r in prior};num=0
 for s in frames:
  item=s['id']+'_'+a.task+('_mirror' if a.mirror else '')+('_27b' if a.model!=CFG['model'] else '')+('_front_first' if a.front_first else '')
  if item in done:continue
  if num>=a.max_new:break
  check_deadline()
  count=len(ledger.read_text().splitlines()) if ledger.exists() else 0
  if count>=CFG['max_vlm_requests_including_failures']:raise RuntimeError('Global VLM request cap reached')
  ip=ROOT/s['file']
  if a.mirror:
   ip=ROOT/'inputs_mirror'/(s['id']+'.png');ip.parent.mkdir(exist_ok=True);ImageOps.mirror(Image.open(ROOT/s['file'])).save(ip)
  prompt=VIEW if a.task=='view' else LOC
  if a.task=='conditioned':
   if s['id'] not in ann_lookup:continue
   x=ann_lookup[s['id']];prompt+='\nVIEW CONTEXT (manual, no leader identity): '+json.dumps({'view':x['view'],'screen_progression_vector':x['direction'],'interpretation':'x right, y down; this is an approximate screen axis, not calibrated track distance. A matching helmet is normally behind its horse nose along this axis.'})
  schema=json.loads(json.dumps(VIEW_SCHEMA if a.task=='view' else LOC_SCHEMA))
  if a.front_first:
   assert a.task=='view';en=schema['properties']['view']['enum'];schema['properties']['view']['enum']=en[1:]+en[:1]
  data=ip.read_bytes();messages=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt,'images':[base64.b64encode(data).decode()]}]
  body={'model':a.model,'messages':messages,'format':schema,'stream':False,'think':False,'keep_alive':'10m','options':CFG['options']};payload=json.dumps(body,ensure_ascii=False).encode()
  import hashlib
  rec={'item_id':item,'frame_id':s['id'],'task':a.task,'mirror':a.mirror,'front_first':a.front_first,'pts':s['actual_pts'],'image':str(ip.relative_to(ROOT)),'image_sha256':sha(ip),'image_count':1,'model':a.model,'model_digest':a.digest,'options':CFG['options'],'schema':schema,'messages':[{'role':m['role'],'content':m['content']} for m in messages],'request_sha256':hashlib.sha256(payload).hexdigest(),'started_at':now(),'attempt':1,'failure':False}
  append(ledger,rec);start=time.perf_counter();raw=None
  try:
   req=urllib.request.Request(CFG['endpoint']+'/api/chat',data=payload,headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=180) as response:raw=response.read().decode();rec['http_status']=response.status
   answer=json.loads(raw);parsed=json.loads(answer['message']['content'])
   assert set(schema['required'])<=set(parsed)
   if a.task!='view':
    assert len(parsed['nose_xy']) in [0,2] and len(parsed['cap_box']) in [0,4]
    if parsed['cap_box']:assert parsed['cap_box'][0]<parsed['cap_box'][2] and parsed['cap_box'][1]<parsed['cap_box'][3]
   rec.update(raw_http_response=raw,response_json=answer,parsed=parsed)
  except Exception as e:
   rec.update(failure=True,error=repr(e))
   if raw is not None:rec['raw_http_response']=raw
  rec['http_elapsed_ms']=(time.perf_counter()-start)*1000;append(out,rec);done.add(item);num+=1
  status('vlm_'+a.task,'running',completed_new=num,last_item=item,failure=rec['failure'],elapsed_ms=rec['http_elapsed_ms'])
  if a.first:print(json.dumps(rec.get('parsed',rec.get('error')),ensure_ascii=False),flush=True)
 status('vlm_'+a.task,'complete',completed_new=num,first_only=a.first)
if __name__=='__main__':main()
