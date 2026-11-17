import argparse,os,time,urllib.request
from common import *
os.environ['TORCH_HOME']=str(ROOT/'model_cache')
import torch,numpy as np
from PIL import Image
from torchvision.models.detection import keypointrcnn_resnet50_fpn,KeypointRCNN_ResNet50_FPN_Weights
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);args=p.parse_args();wait_gpu();torch.set_num_threads(4)
 weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1;url=weights.url;target=ROOT/'model_cache/checkpoints'/url.rsplit('/',1)[-1]
 if not target.exists():
  guard()
  with urllib.request.urlopen(url,timeout=60) as r:
   size=int(r.headers.get('Content-Length','0'));assert 0<size<=CFG['max_new_weight_bytes'];tmp=target.with_suffix('.part');total=0
   with tmp.open('wb') as f:
    for chunk in iter(lambda:r.read(1<<20),b''):
     total+=len(chunk);assert total<=CFG['max_new_weight_bytes'];f.write(chunk)
  assert total==size;os.replace(tmp,target)
 dump(ROOT/'evidence/pose_weights.json',{'url':url,'file':str(target.relative_to(ROOT)),'bytes':target.stat().st_size,'sha256':sha(target),'enum':str(weights),'keypoint_names':weights.meta['keypoint_names'],'torchvision_installed':__import__('torchvision').__version__,'documentation':'https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html'})
 model=keypointrcnn_resnet50_fpn(weights=weights).eval().cuda();out=ROOT/'raw/pose.jsonl';done={r['frame'] for r in readlines(out)}
 frames=[f for f in json.loads((ROOT/'frame_manifest.json').read_text()) if f['split'] in args.split.split(',')]
 for f in frames:
  if f['id'] in done:continue
  guard();im=Image.open(ROOT/f['file']).convert('RGB');t=torch.from_numpy(np.array(im)).permute(2,0,1).float().cuda()/255;count_inference('keypointrcnn',f['id']);torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():r=model([t])[0]
  torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000;poses=[]
  for j,s in enumerate(r['scores']):
   if float(s)>=.5:poses.append({k:r[k][j].detach().cpu().tolist() for k in ['boxes','scores','keypoints','keypoints_scores']})
  poses=[{'box':q['boxes'],'score':q['scores'],'keypoints':q['keypoints'],'keypoints_scores':q['keypoints_scores']} for q in poses]
  append(out,{'frame':f['id'],'file':f['file'],'image_sha256':sha(ROOT/f['file']),'poses':poses,'forward_ms':ms});status('C_pose','running',frame=f['id'],persons=len(poses),forward_ms=ms)
 status('C_pose','complete',split=args.split,frames=len(frames))
if __name__=='__main__':main()
