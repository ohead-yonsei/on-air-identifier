"""Official generic horse/person detector diagnostic; no training or nose detector."""
import argparse,hashlib,json,os,platform,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,CFG,sha,dump,append,status,check_deadline

def main():
 p=argparse.ArgumentParser();p.add_argument('--first',action='store_true');a=p.parse_args()
 import torch,torchvision
 from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
 torch.set_num_threads(4);torch.hub.set_dir(str(ROOT/'model_cache'))
 w=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
 status('detector','loading',weights=str(w),cache=str(ROOT/'model_cache'))
 model=fasterrcnn_resnet50_fpn_v2(weights=w,progress=False).eval().to('cuda')
 cached=list((ROOT/'model_cache/checkpoints').glob('*.pth'));assert sum(p.stat().st_size for p in cached)<=CFG['max_download_bytes']
 dump(ROOT/'evidence/detector_environment.json',{'time':time.time(),'torch':torch.__version__,'torchvision':torchvision.__version__,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(),'model':'fasterrcnn_resnet50_fpn_v2','weights':str(w),'weights_url':w.url,'weight_files':[{'path':str(x.relative_to(ROOT)),'bytes':x.stat().st_size,'sha256':sha(x)} for x in cached],'categories':{'1':'person','19':'horse'},'input':'single masked1280x720 RGB; model default resize; float32; eval/inference_mode','scope':'Generic COCO boxes; not trained jockey/helmet/nose detections.','training_steps':0})
 selection=json.loads((ROOT/'selection.json').read_text())['frames']
 if a.first:selection=[next(s for s in selection if s['target_seconds']==35)]
 out=ROOT/'raw/detector.jsonl';done={json.loads(l)['id'] for l in out.read_text().splitlines()} if out.exists() else set()
 for s in selection:
  if s['id'] in done:continue
  check_deadline();assert len(done)<CFG['max_detector_frames']
  im=Image.open(ROOT/s['file']).convert('RGB');arr=np.asarray(im).copy();x=torch.from_numpy(arr).permute(2,0,1).float().div_(255).to('cuda')
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():r=model([x])[0]
  torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
  det=[]
  for box,lab,score in zip(r['boxes'].cpu().tolist(),r['labels'].cpu().tolist(),r['scores'].cpu().tolist()):
   if lab in [1,19] and score>=0.25:det.append({'box':box,'label':lab,'class':'person' if lab==1 else 'horse','score':score})
  rec={'id':s['id'],'pts':s['actual_pts'],'image':s['file'],'image_sha256':sha(ROOT/s['file']),'elapsed_forward_ms':elapsed,'detections':det,'failure':False};append(out,rec);done.add(s['id'])
  board=im.copy();draw=ImageDraw.Draw(board);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
  for d in det:
   if d['score']<0.5:continue
   b=d['box'];col='#46ff89' if d['label']==19 else '#ffcb42';draw.rectangle(b,outline=col,width=2);draw.text((b[0],max(0,b[1]-16)),f'{d["class"]} {d["score"]:.2f}',font=font,fill=col,stroke_width=2,stroke_fill='black')
  dest=ROOT/'review/detector';dest.mkdir(exist_ok=True);board.save(dest/(s['id']+'.jpg'),quality=95)
  status('detector','running',completed=len(done),last_id=s['id'],forward_ms=elapsed,horse_boxes=sum(d['label']==19 and d['score']>=0.5 for d in det),person_boxes=sum(d['label']==1 and d['score']>=0.5 for d in det))
 status('detector','complete',completed=len(done),first_only=a.first)
if __name__=='__main__':main()
