import argparse,base64,hashlib,json,math,time,urllib.request
import cv2,numpy as np
from PIL import Image,ImageOps
from common import *
from regions import *
import v4_color as v4

def hsv_classify(im,box,person,width):
 start=time.perf_counter()
 if box is None or person is None:return {'number':0,'state':'unknown','reason':'no_region_or_person','elapsed_ms':0}
 h,w=im.shape[:2];b=clip(person,w,h);l,t,r,d=b;body=im[t:d,l:r];lab=cv2.cvtColor(body,cv2.COLOR_BGR2LAB).astype(np.float32);bg=np.median(np.concatenate([lab[0],lab[:,0],lab[:,-1]]),axis=0)
 l,t,r,d=box;cap=im[t:d,l:r];fg=(np.linalg.norm(cv2.cvtColor(cap,cv2.COLOR_BGR2LAB).astype(np.float32)-bg,axis=2)>20).reshape(-1).astype(float)
 ch,known=v4.hist(v4.quant(np.ascontiguousarray(cap)),fg+1e-9);scores=[v4.cap_score(ch,n)*min(1.,known*4) for n in range(1,12)];rank=np.argsort(scores)[::-1];best=float(scores[rank[0]]);margin=best-float(scores[rank[1]]);unknown=best<.25 or margin<.08;high=not unknown and best>=.40 and margin>=.08 and known>=.25 and person[3]-person[1]>=60*width/1280
 return {'number':0 if unknown else int(rank[0])+1,'top_candidate':int(rank[0])+1,'state':'unknown' if unknown else ('high_evidence' if high else 'guess'),'score':best,'margin':margin,'known_fraction':float(known),'histogram':ch.tolist(),'all_scores':scores,'elapsed_ms':(time.perf_counter()-start)*1000}

def request(item,image,keep_alive='10m'):
 protocol=json.loads((ROOT/'protocols/cap_classifier.json').read_text());prior=readlines(ROOT/'raw/vlm.jsonl');old=[r for r in prior if r['item_id']==item]
 if old and not old[-1]['failure']:return old[-1]
 attempts=len(old)
 if attempts>=2:return old[-1]
 for attempt in range(attempts+1,3):
  guard();ledger=ROOT/'raw/request_ledger.jsonl';assert len(readlines(ledger))<CFG['max_vlm_requests'],'VLM cap reached'
  msgs=[{'role':'system','content':protocol['SYSTEM']},{'role':'user','content':protocol['PROMPT'],'images':[base64.b64encode(image.read_bytes()).decode()]}];payload={'model':CFG['model'],'messages':msgs,'format':protocol['SCHEMA'],'stream':False,'think':False,'keep_alive':keep_alive,'options':protocol['options']};body=json.dumps(payload).encode()
  rec={'item_id':item,'attempt':attempt,'time_utc':now(),'model':CFG['model'],'model_digest':CFG['model_digest'],'image':str(image.relative_to(ROOT)),'image_sha256':sha(image),'messages':[{'role':m['role'],'content':m['content']} for m in msgs],'schema':protocol['SCHEMA'],'options':protocol['options'],'think':False,'stream':False,'keep_alive':keep_alive,'request_sha256':hashlib.sha256(body).hexdigest()};append(ledger,rec);raw=None;start=time.perf_counter()
  try:
   with urllib.request.urlopen(urllib.request.Request(CFG['endpoint']+'/api/chat',data=body,headers={'Content-Type':'application/json'}),timeout=180) as r:raw=r.read().decode();rec['http_status']=r.status
   parsed=json.loads(json.loads(raw)['message']['content']);assert set(parsed)=={'color_class'} and parsed['color_class'] in protocol['CLASSES'];rec.update(failure=False,parsed=parsed,number=protocol['CLASSES'].index(parsed['color_class']))
  except Exception as e:rec.update(failure=True,error=repr(e))
  rec.update(raw_http_response=raw,http_elapsed_ms=(time.perf_counter()-start)*1000);append(ROOT/'raw/vlm.jsonl',rec)
  if not rec['failure']:return rec
  recent=readlines(ROOT/'raw/vlm.jsonl')[-3:]
  if len(recent)==3 and all(x['failure'] for x in recent):raise RuntimeError('three consecutive inference errors')
 return rec

def quality(box,ref,face,torso):
 c=[(ref[0]+ref[2])/2,(ref[1]+ref[3])/2];ar=area(box)
 return {'cap_iou':iou(box,ref),'cap_center_included':bool(box and box[0]<=c[0]<box[2] and box[1]<=c[1]<box[3]),'outside_cap_box_fraction':1-inter(box,ref)/ar if ar else None,'face_box_overlap_fraction':inter(box,face)/ar if ar and face is not None else None,'torso_box_overlap_fraction':inter(box,torso)/ar if ar and torso is not None else None,'contamination_definition':'overlap of draft rectangles, not pixel segmentation'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',required=True);ap.add_argument('--first',action='store_true');a=ap.parse_args();cv2.setNumThreads(4)
 lock=json.loads((ROOT/'evidence/annotation_lock.json').read_text());assert sha(ROOT/'annotations.json')==lock['sha256']
 fm={f['id']:f for f in json.loads((ROOT/'frame_manifest.json').read_text())};det={r['frame']:r for r in readlines(ROOT/'raw/detector.jsonl')};poses={r['frame']:r for r in readlines(ROOT/'raw/pose.jsonl')};samples=[s for s in json.loads((ROOT/'annotations.json').read_text())['samples'] if s['split'] in a.split.split(',')];rows=[];images={};generation=[]
 for s in samples:
  f=fm[s['frame']];im=cv2.imread(str(ROOT/f['file']));people=[q for q in det[s['frame']]['detections'] if q['class']=='person'];person=next((q for q in people if q['id']==s.get('person_id')),None);b=person['box'] if person else None;pm=pose_match(people,poses.get(s['frame'],{}).get('poses',[]))
  for method in CFG['methods']:
   item=s['id']+'_'+method;box,info,ms=region(im,b,method,pm.get(s.get('person_id')),s['cap_box']);row={'item_id':item,'sample_id':s['id'],'race':s['race'],'frame':s['frame'],'split':s['split'],'view':s['view'],'reference_number':s['number'],'reference_status':s['label_status'],'person_id':s.get('person_id'),'person_box':b,'method':method,'region_box':box,'region_info':info,'region_ms':ms,'size_group':('missing' if b is None else 'small' if b[3]-b[1]<60*f['width']/1280 else 'medium' if b[3]-b[1]<120*f['width']/1280 else 'large')};row.update(quality(box,s['cap_box'],s.get('face_box'),s.get('torso_box')));row['hsv']=hsv_classify(im,box,b,f['width'])
   if box is not None:
    rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));crop=rgb.crop(box);can=Image.new('RGB',(224,224),(96,96,96));patch=ImageOps.contain(crop,(224,224),Image.Resampling.LANCZOS);can.paste(patch,((224-patch.width)//2,(224-patch.height)//2));ip=ROOT/'inputs'/(item+'.png');can.save(ip);row.update(image=str(ip.relative_to(ROOT)),image_sha256=sha(ip),native_size=list(crop.size));generation.append((item,ip));images[item]=ip
   else:row['vlm']={'number':0,'state':'region_unavailable','http_elapsed_ms':0,'failure':False}
   rows.append(row)
 dest=ROOT/'results'/('regions_'+a.split.replace(',','_')+'.json');dump(dest,rows)
 for i,(item,ip) in enumerate(generation):
  r=request(item,ip,keep_alive=0 if (a.first or i==len(generation)-1) else '10m');row=next(q for q in rows if q['item_id']==item);row['vlm']={'number':r.get('number',0),'failure':r['failure'],'http_elapsed_ms':r['http_elapsed_ms'],'state':'request_failure' if r['failure'] else ('unknown' if r['number']==0 else 'asserted_class'),'request_attempt':r['attempt']};dump(dest,rows);status('C_classification','running',split=a.split,item=item,completed=i+1,total=len(generation))
  if a.first:break
 status('C_classification','first_checked_pending_review' if a.first else 'complete',split=a.split,items=len(rows),actual_requests=len(readlines(ROOT/'raw/vlm.jsonl')))
if __name__=='__main__':main()
