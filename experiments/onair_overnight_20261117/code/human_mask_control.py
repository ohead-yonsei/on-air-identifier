"""Post-result protocol correction: mask the single human-check source before detection.
Preserve all primary results. Reuse VLM responses only for byte-identical inputs/protocol.
No parameter tuning or new labels. New raw records and request ledger remain separate.
"""
import os,time,json,gc
from common import *
os.environ['TORCH_HOME']=str(ROOT/'model_cache')
import cv2,numpy as np,torch
from PIL import Image,ImageOps
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights,keypointrcnn_resnet50_fpn,KeypointRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms
from regions import iou,pose_match,region
from classify_regions import request,hsv_classify,quality

def main():
 assert not (ROOT/'results/human_mask_control.json').exists(),'preserve previous control'
 wait_gpu();torch.set_num_threads(4);cv2.setNumThreads(4);start_calls=len(readlines(ROOT/'raw/vlm.jsonl'))
 ann=[s for s in json.loads((ROOT/'annotations.json').read_text())['samples'] if s['split']=='human_check'];src=ROOT/'frames/HUMAN_t038_5.png';im=cv2.imread(str(src));masks=[[0,0,240,110],[0,610,1280,720]]
 for l,t,r,b in masks:im[t:b,l:r]=48
 path=ROOT/'frames/HUMAN_t038_5_masked_control.png';cv2.imwrite(str(path),im);tensor=torch.from_numpy(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float().cuda()/255
 dump(ROOT/'evidence/human_mask_control_input.json',{'time_utc':now(),'source':str(src.relative_to(ROOT)),'source_sha256':sha(src),'file':str(path.relative_to(ROOT)),'sha256':sha(path),'masks':masks,'pts':38.5,'purpose':'Input protocol correction after final visual audit; no change to original frozen experiment.'})
 people=[];poses=[]
 for kind in ['detector','pose']:
  model=(fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1) if kind=='detector' else keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)).eval().cuda();count_inference('human_mask_'+kind,'HUMAN_t038_5_masked_control');torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():pred=model([tensor])[0]
  torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
  if kind=='detector':
   keep=(pred['labels']==1)&(pred['scores']>=.5);boxes=pred['boxes'][keep];scores=pred['scores'][keep];ix=nms(boxes,scores,.45);pairs=sorted(zip(boxes[ix].cpu().tolist(),scores[ix].cpu().tolist()),key=lambda x:(x[0][0]+x[0][2])/2);people=[{'id':'p'+str(j),'class':'person','box':box,'score':score} for j,(box,score) in enumerate(pairs)];record={'detections':people}
  else:
   poses=[{'box':pred['boxes'][j].cpu().tolist(),'score':float(v),'keypoints':pred['keypoints'][j].cpu().tolist(),'keypoints_scores':pred['keypoints_scores'][j].cpu().tolist()} for j,v in enumerate(pred['scores']) if float(v)>=.5];record={'poses':poses}
  append(ROOT/'raw/human_mask_vision.jsonl',{'kind':kind,'file':str(path.relative_to(ROOT)),'image_sha256':sha(path),'forward_ms':ms,**record});del pred,model;gc.collect();torch.cuda.empty_cache()
 # Preserve identity correspondence through original person boxes, not through predicted colour.
 cost=np.full((len(ann),len(people)+len(ann)),.5)
 for i,s in enumerate(ann):
  for j,p in enumerate(people):cost[i,j]=1-iou(s['person_box'],p['box']) if iou(s['person_box'],p['box'])>=.5 else 10
 ix,jx=linear_sum_assignment(cost);matched={ann[i]['id']:people[j] for i,j in zip(ix,jx) if j<len(people) and cost[i,j]<=.5};pm=pose_match(people,poses);rows=[];reuse={r['image_sha256']:r for r in readlines(ROOT/'raw/vlm.jsonl') if not r['failure']}
 for s in ann:
  person=matched.get(s['id']);b=person['box'] if person else None
  for method in CFG['methods']:
   item=s['id']+'_'+method+'_masked_control';box,info,ms=region(im,b,method,pm.get(person['id']) if person else None,s['cap_box']);row={'item_id':item,'sample_id':s['id'],'method':method,'reference_number':s['number'],'reference_status':'human_confirmed_colour_only','person_box':b,'original_person_box':s['person_box'],'person_match_iou':iou(b,s['person_box']),'region_box':box,'region_info':info,'region_ms':ms,'hsv':hsv_classify(im,box,b,1280)};row.update(quality(box,s['cap_box'],s.get('face_box'),s.get('torso_box')))
   if box is None:row['vlm']={'number':0,'failure':False,'state':'region_unavailable','http_elapsed_ms':0}
   else:
    crop=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).crop(box);can=Image.new('RGB',(224,224),(96,96,96));part=ImageOps.contain(crop,(224,224),Image.Resampling.LANCZOS);can.paste(part,((224-part.width)//2,(224-part.height)//2));dest=ROOT/'inputs'/(item+'.png');can.save(dest);h=sha(dest);row.update(image=str(dest.relative_to(ROOT)),image_sha256=h,native_size=list(crop.size))
    if h in reuse:
     old=reuse[h];row['vlm']={'number':old['number'],'failure':False,'http_elapsed_ms':None,'state':'reused_byte_identical_input','reused_item_id':old['item_id']};append(ROOT/'raw/human_mask_reuse.jsonl',{'time_utc':now(),'item_id':item,'reused_item_id':old['item_id'],'image_sha256':h,'reason':'byte-identical crop and unchanged protocol; no new request'})
   rows.append(row)
 calls=[r for r in rows if 'vlm' not in r]
 for i,r in enumerate(calls):
  out=request(r['item_id'],ROOT/r['image'],keep_alive=0 if i==len(calls)-1 else '10m');r['vlm']={'number':out.get('number',0),'failure':out['failure'],'http_elapsed_ms':out['http_elapsed_ms'],'state':'new_request'}
 dump(ROOT/'results/human_mask_control.json',rows);summary={}
 for m in CFG['methods']:
  group=[r for r in rows if r['method']==m];summary[m]={}
  for cls in ['vlm','hsv']:summary[m][cls]={'correct':sum(r[cls]['number']==r['reference_number'] for r in group),'wrong':sum(r[cls]['number'] not in [0,r['reference_number']] for r in group),'abstain':sum(r[cls]['number']==0 for r in group)}
 dump(ROOT/'results/human_mask_summary.json',{'methods':summary,'observations':6,'matched_people':len(matched),'conditions':30,'new_vlm_requests':len(readlines(ROOT/'raw/vlm.jsonl'))-start_calls,'reused_requests':sum(r['vlm']['state']=='reused_byte_identical_input' for r in rows),'unavailable_regions':sum(r['region_box'] is None for r in rows),'status':'post-result input correction; original primary results preserved; no threshold/label edits'})
 status('human_mask_control','complete',conditions=30,new_requests=len(calls),total_requests=len(readlines(ROOT/'raw/vlm.jsonl')))
if __name__=='__main__':main()
