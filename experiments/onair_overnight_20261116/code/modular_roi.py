"""Post-hoc modular diagnostic. ROI creation never uses reference cap/nose coords."""
import base64,csv,hashlib,json,math,time,urllib.request
from types import SimpleNamespace
from PIL import Image,ImageOps,ImageDraw,ImageFont
from common import ROOT,CFG,sha,dump,append,status,check_deadline,now

DIR={'left':[-1,0],'down_left':[-1,1],'down':[0,1],'down_right':[1,1],'right':[1,0],'up_right':[1,-1],'up':[0,-1],'up_left':[-1,-1]}
def main():
 protocol=json.loads((ROOT/'protocols/cap_classifier.json').read_text());mod=SimpleNamespace(**protocol)
 opts=protocol['options'];ann=json.loads((ROOT/'geometry_annotations.json').read_text())
 det={r['id']:r for r in map(json.loads,(ROOT/'raw/detector.jsonl').read_text().splitlines())};vlm={r['item_id']:r for r in map(json.loads,(ROOT/'raw/vlm.jsonl').read_text().splitlines())}
 out=ROOT/'raw/modular_cap.jsonl';prior=[json.loads(l) for l in out.read_text().splitlines()] if out.exists() else [];cache={r['image_sha256']:r for r in prior if not r['failure']};rows=[]
 dest=ROOT/'modular';dest.mkdir(exist_ok=True)
 for f in ann['frames']:
  for mode in ['manual_axis','model9b_axis']:
   axis=f['direction'] if mode=='manual_axis' else DIR.get(vlm[f['id']+'_view']['parsed']['direction'])
   item=f['id']+'_'+mode;row={'item_id':item,'frame':f['id'],'axis_source':mode,'axis':axis,'reference_status':'new frame visual draft; manual-axis arm is not fully automatic'}
   if axis is None:row['status']='unknown_axis';rows.append(row);continue
   norm=math.hypot(*axis);axis=[v/norm for v in axis]
   horses=[d for d in det[f['id']]['detections'] if d['class']=='horse' and d['score']>=.5];people=[d for d in det[f['id']]['detections'] if d['class']=='person' and d['score']>=.5]
   if not horses:row['status']='no_horse';rows.append(row);continue
   h=max(horses,key=lambda d:((d['box'][0]+d['box'][2])/2)*axis[0]+((d['box'][1]+d['box'][3])/2)*axis[1]);l,t,r,b=h['box'];w=r-l;hh=b-t;cx=(l+r)/2;choices=[]
   for p in people:
    pl,pt,pr,pb=p['box'];px=(pl+pr)/2;py=(pt+pb)/2
    if l-.2*w<=px<=r+.2*w and t-.75*hh<=py<=b-.2*hh:
     cost=math.hypot((px-cx)/w,(py-(t+.1*hh))/hh)-.05*p['score'];choices.append((cost,p))
   row.update(selected_horse=h,person_candidates=len(choices))
   if not choices:row['status']='no_matching_person';rows.append(row);continue
   _,person=min(choices,key=lambda p:p[0]);pl,pt,pr,pb=person['box'];box=[max(0,int(pl)),max(0,int(pt)),min(1280,math.ceil(pr)),min(720,math.ceil(pt+.35*(pb-pt)))];row.update(selected_person=person,helmet_roi=box)
   im=Image.open(ROOT/f['file']).convert('RGB');crop=im.crop(box);can=Image.new('RGB',(224,224),(96,96,96));patch=ImageOps.contain(crop,(224,224),Image.Resampling.LANCZOS);can.paste(patch,((224-patch.width)//2,(224-patch.height)//2));ip=dest/(item+'.png');can.save(ip);ih=sha(ip);row.update(image=str(ip.relative_to(ROOT)),image_sha256=ih)
   if ih in cache:rec=cache[ih];row['classification_reused_from']=rec['item_id']
   else:
    check_deadline();ledger=ROOT/'raw/request_ledger.jsonl';assert len(ledger.read_text().splitlines())<CFG['max_vlm_requests_including_failures']
    payload={'model':CFG['model'],'messages':[{'role':'system','content':mod.SYSTEM},{'role':'user','content':mod.PROMPT,'images':[base64.b64encode(ip.read_bytes()).decode()]}],'format':mod.SCHEMA,'stream':False,'think':False,'keep_alive':'10m','options':opts};body=json.dumps(payload).encode()
    rec={'item_id':item,'stage':'modular_cap','model':CFG['model'],'model_digest':CFG['model_digest'],'options':opts,'messages':[{'role':m['role'],'content':m['content']} for m in payload['messages']],'schema':mod.SCHEMA,'image':str(ip.relative_to(ROOT)),'image_sha256':ih,'request_sha256':hashlib.sha256(body).hexdigest(),'started_at':now(),'attempt':1,'failure':False}
    append(ledger,rec);s=time.perf_counter();raw=None
    try:
     req=urllib.request.Request(CFG['endpoint']+'/api/chat',data=body,headers={'Content-Type':'application/json'})
     with urllib.request.urlopen(req,timeout=180) as rr:raw=rr.read().decode();rec['http_status']=rr.status
     parsed=json.loads(json.loads(raw)['message']['content']);assert parsed['color_class'] in mod.CLASSES
     rec.update(raw_http_response=raw,parsed=parsed,number=mod.CLASSES.index(parsed['color_class']))
    except Exception as e:rec.update(failure=True,error=repr(e),raw_http_response=raw)
    rec['http_elapsed_ms']=(time.perf_counter()-s)*1000;append(out,rec)
    if not rec['failure']:cache[ih]=rec
   row.update(status='classification_failure' if rec['failure'] else 'ok',predicted_number=rec.get('number'),parsed=rec.get('parsed'),classification_http_ms=rec['http_elapsed_ms'])
   # Evaluation only below this line; references did not influence ROI selection.
   gold=next(n for n in f['noses'] if n['reference_number']==f['leader_reference_number']);n=gold['point'];c=next(c for c in f['caps'] if c['id']==gold['reference_cap_id'])['center']
   row['selected_horse_contains_reference_leader_nose']=l<=n[0]<=r and t<=n[1]<=b;row['roi_contains_reference_leader_cap_center']=box[0]<=c[0]<=box[2] and box[1]<=c[1]<=box[3];row['number_matches_reference_leader']=rec.get('number')==f['leader_reference_number']
   draw=ImageDraw.Draw(im);draw.rectangle(h['box'],outline='#40ff70',width=3);draw.rectangle(person['box'],outline='#ffcf40',width=2);draw.rectangle(box,outline='#50d8ff',width=3);draw.text((250,20),f'{item} cap prediction={rec.get("number")}',font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22),fill='white',stroke_fill='black',stroke_width=2);im.save(dest/(item+'_overlay.jpg'),quality=95)
   rows.append(row);status('modular_roi','running',item=item,classification_reused='classification_reused_from' in row)
 dump(ROOT/'results/modular_pipeline.json',rows)
 status('modular_roi','complete',pipeline_cases=len(rows),unique_new_calls=len([r for r in map(json.loads,out.read_text().splitlines())]))
if __name__=='__main__':main()
