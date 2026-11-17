import argparse,os,time,json,math
from common import *
os.environ['TORCH_HOME']=str(ROOT/'model_cache')
import cv2,numpy as np,torch,torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.ops import nms
from PIL import Image,ImageDraw,ImageFont,ImageOps
p=argparse.ArgumentParser();p.add_argument('--manifest',default='frame_manifest.json');p.add_argument('--out',default='detector');a=p.parse_args()
torch.set_num_threads(4);cv2.setNumThreads(4);wait_gpu()
weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1;model=fasterrcnn_resnet50_fpn_v2(weights=weights).eval().cuda()
manifest=json.loads((ROOT/a.manifest).read_text());out=ROOT/'raw'/(a.out+'.jsonl');done={r['frame'] for r in readlines(out)}
dest=ROOT/'review/riders';dest.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
for f in manifest:
 if f['id'] in done:continue
 guard();im=Image.open(ROOT/f['file']).convert('RGB');arr=np.asarray(im);tensor=torch.from_numpy(arr.copy()).permute(2,0,1).float().cuda()/255
 count_inference('fasterrcnn',f['id']);torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode():pred=model([tensor])[0]
 torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000;ds=[]
 for cls,name in [(1,'person'),(19,'horse')]:
  mask=(pred['labels']==cls)&(pred['scores']>=.5);boxes=pred['boxes'][mask];scores=pred['scores'][mask];keep=nms(boxes,scores,.45)
  pairs=sorted(zip(boxes[keep].cpu().tolist(),scores[keep].cpu().tolist()),key=lambda x:(x[0][0]+x[0][2])/2)
  for j,(box,score) in enumerate(pairs):ds.append({'id':name[0]+str(j),'class':name,'box':box,'score':score})
 rec={'frame':f['id'],'file':f['file'],'image_sha256':sha(ROOT/f['file']),'width':im.width,'height':im.height,'detections':ds,'forward_ms':ms,'weights':str(weights),'detector_input_min_size':model.transform.min_size,'max_size':model.transform.max_size};append(out,rec)
 if a.out=='detector':
  overlay=im.copy();d=ImageDraw.Draw(overlay);tiles=[];tilemeta=[]
  for q in ds:
   if q['class']!='person':continue
   l,t,r,b=q['box'];d.rectangle(q['box'],outline='#00ff90',width=2);d.text((l,max(0,t-20)),q['id'],font=font,fill='yellow',stroke_fill='black',stroke_width=2)
   box=[max(0,math.floor(l-5)),max(0,math.floor(t-5)),min(im.width,math.ceil(r+5)),min(im.height,math.ceil(t+.5*(b-t)+5))];patch=im.crop(box);scale=min(300/patch.width,240/patch.height);nw,nh=round(patch.width*scale),round(patch.height*scale);tile=Image.new('RGB',(320,290),'#20313c');tile.paste(patch.resize((nw,nh)),(10,42));td=ImageDraw.Draw(tile);td.text((8,3),f'{q["id"]} origin({box[0]},{box[1]}) x{scale:.2f}',font=small,fill='white');td.text((8,21),f'box {list(map(lambda x:round(x,1),q["box"]))}',font=small,fill='white')
   # Grid in patch-native coordinates, outside text areas.
   for x in range(0,patch.width,10):
    xx=10+x*scale;td.line((xx,42,xx,42+nh),fill=(100,100,100),width=1);td.text((xx,42),str(x),font=small,fill='#ffff00',stroke_width=1,stroke_fill='black')
   for y in range(0,patch.height,10):
    yy=42+y*scale;td.line((10,yy,10+nw,yy),fill=(100,100,100),width=1);td.text((10,yy),str(y),font=small,fill='#ffff00',stroke_width=1,stroke_fill='black')
   tiles.append(tile);tilemeta.append({'person_id':q['id'],'crop_box':box,'scale':scale})
  overlay.save(dest/(f['id']+'_boxes.jpg'),quality=95)
  if tiles:
   sheet=Image.new('RGB',(1280,290*math.ceil(len(tiles)/4)),(30,30,30))
   for k,tile in enumerate(tiles):sheet.paste(tile,((k%4)*320,(k//4)*290))
   sheet.save(dest/(f['id']+'_candidates.jpg'),quality=95)
  dump(dest/(f['id']+'_tiles.json'),tilemeta)
 status('detector_'+a.out,'running',frame=f['id'],persons=sum(q['class']=='person' for q in ds),horses=sum(q['class']=='horse' for q in ds),forward_ms=ms)
status('detector_'+a.out,'complete',frames=len(readlines(out)))
