"""Only the extra experiments named before classification in PLAN_KO.md."""
import collections,json,math,time
import cv2,numpy as np
from PIL import Image,ImageOps
from common import *
from regions import region,clip,iou
from classify_regions import hsv_classify,request
def main():
 cv2.setNumThreads(4);ann=json.loads((ROOT/'annotations.json').read_text())['samples'];fm={f['id']:f for f in json.loads((ROOT/'frame_manifest.json').read_text())};chosen=[]
 for race in CFG['evaluation_races']:
  frames=sorted([f for f in fm.values() if f['race']==race],key=lambda f:f['actual_pts']);first=frames[0]['id'];chosen+=sorted([s for s in ann if s['frame']==first],key=lambda s:s['id'])[:2]
 assert len(chosen)==12
 dump(ROOT/'evidence/resolution_selection.json',{'time_utc':now(),'selection_rule':'earliest PTS frame per evaluation race, first two pre-existing sample IDs; no outcome-based selection','samples':[s['id'] for s in chosen],'widths':[768,1280],'baseline_width':1920,'coordinates':'scale original manual U cap and common person boxes; no redraw'})
 rows=[];calls=[]
 for s in chosen:
  f=fm[s['frame']];im=cv2.imread(str(ROOT/f['file']))
  for width in [768,1280]:
   scale=width/f['width'];small=cv2.resize(im,(width,round(f['height']*scale)),interpolation=cv2.INTER_AREA);box=clip([v*scale for v in s['cap_box']],small.shape[1],small.shape[0]);person=[v*scale for v in s['person_box']];item=s['id']+f'_U_w{width}';crop=Image.fromarray(cv2.cvtColor(small,cv2.COLOR_BGR2RGB)).crop(box);can=Image.new('RGB',(224,224),(96,96,96));part=ImageOps.contain(crop,(224,224),Image.Resampling.LANCZOS);can.paste(part,((224-part.width)//2,(224-part.height)//2));dest=ROOT/'inputs'/f'{item}.png';can.save(dest)
   row={'item_id':item,'sample_id':s['id'],'race':s['race'],'frame':s['frame'],'width':width,'scale':scale,'cap_box':box,'native_cap_size':list(crop.size),'reference_number':s['number'],'reference_status':s['label_status'],'image':str(dest.relative_to(ROOT)),'image_sha256':sha(dest),'hsv':hsv_classify(small,box,person,width)};rows.append(row);calls.append((item,dest,row))
 for i,(item,img,row) in enumerate(calls):
  r=request(item,img,keep_alive=0 if i==len(calls)-1 else '10m');row['vlm']={'number':r.get('number',0),'failure':r['failure'],'http_elapsed_ms':r['http_elapsed_ms']};dump(ROOT/'results/resolution_extra.json',rows);status('extra_resolution','running',item=item,completed=i+1,total=len(calls))
 base={r['sample_id']:r for p in (ROOT/'results').glob('regions_*.json') for r in json.loads(p.read_text()) if r['method']=='R2_cluster'};jitter=[]
 for s in ann:
  if s['split']!='development':continue
  f=fm[s['frame']];im=cv2.imread(str(ROOT/f['file']));b=s['person_box'];ref=base[s['id']]
  for dx,dy in [(-2,0),(2,0),(0,-2),(0,2)]:
   shifted=[b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy];box,info,ms=region(im,shifted,'R2_cluster');pred=hsv_classify(im,box,shifted,f['width']);jitter.append({'sample_id':s['id'],'race':s['race'],'dx':dx,'dy':dy,'region_box':box,'region_ms':ms,'baseline_region':ref['region_box'],'iou_with_baseline':iou(box,ref['region_box']),'baseline_number':ref['hsv']['number'],'new_number':pred['number'],'changed_number':ref['hsv']['number']!=pred['number'],'reference_number':s['number'],'region_failed':box is None})
 dump(ROOT/'results/R2_jitter_extra.json',jitter);status('extra_R2_jitter','complete',rows=len(jitter),changed_predictions=sum(x['changed_number'] for x in jitter))
 status('extra_resolution','complete',requests=len(calls),total_vlm_requests=len(readlines(ROOT/'raw/vlm.jsonl')))
if __name__=='__main__':main()
