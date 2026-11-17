"""Causal .5s sparse flow with background homography compensation.
No manual direction, cap labels, or reference nose coordinates enter inference.
Thresholds below are fixed before the first run; outputs may abstain.
"""
import argparse,bisect,json,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,CFG,dump,append,status,sha

CONFIG={'lookback_seconds':.5,'background_corners':500,'foreground_corners_union_mask':800,'quality_level':.01,'min_distance':4,'lk_window':31,'lk_pyramid_levels':4,'forward_backward_max_error_px':1.5,'homography_ransac_threshold_px':3.0,'min_background_inliers':20,'min_foreground_residual_px':1.0,'min_foreground_tracks':10,'min_direction_concentration':.5,'detector_box_threshold':.5,'source':'automatic generic horse boxes and two past/current frames only; no manual direction or identity input','scope':'exploratory direction estimate; background motion compensation is not metric track calibration'}

def tracked(current,previous,mask,max_corners):
 p=cv2.goodFeaturesToTrack(current,maxCorners=max_corners,qualityLevel=.01,minDistance=4,mask=mask)
 if p is None:return np.empty((0,2)),np.empty((0,2))
 params={'winSize':(31,31),'maxLevel':4,'criteria':(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01)}
 prev,ok,_=cv2.calcOpticalFlowPyrLK(current,previous,p,None,**params)
 back,ok2,_=cv2.calcOpticalFlowPyrLK(previous,current,prev,None,**params)
 valid=(ok.reshape(-1)>0)&(ok2.reshape(-1)>0)&(np.linalg.norm(back.reshape(-1,2)-p.reshape(-1,2),axis=1)<=1.5)
 return prev.reshape(-1,2)[valid],p.reshape(-1,2)[valid]

def main():
 p=argparse.ArgumentParser();p.add_argument('--lookback',type=float,default=.5);p.add_argument('--motion-model',choices=['homography','affine'],default='homography');p.add_argument('--tag',default='flow');p.add_argument('--frames',default='');p.add_argument('--mirror',action='store_true');a=p.parse_args()
 result=ROOT/'results'/(a.tag+'_direction.json')
 if result.exists():status('flow_direction','skip_existing',result=str(result));return
 config={**CONFIG,'lookback_seconds':a.lookback,'motion_model':a.motion_model,'tag':a.tag,'frame_ids_override':a.frames,'posthoc':a.tag!='flow','mirror':a.mirror}
 cv2.setNumThreads(4);dest=ROOT/a.tag;dest.mkdir(exist_ok=True);dump(dest/'config.json',config)
 frames=[f for f in json.loads((ROOT/'selection.json').read_text())['frames'] if f['geometry_review']]
 if a.frames:frames=[f for f in json.loads((ROOT/'selection.json').read_text())['frames'] if f['id'] in a.frames.split(',')]
 det={r['id']:r for r in map(json.loads,(ROOT/'raw/detector.jsonl').read_text().splitlines())};pts=json.loads((ROOT/'evidence/video_pts.json').read_text())
 targets={bisect.bisect_right(pts,f['actual_pts']-a.lookback+1e-8)-1:f['id'] for f in frames};cap=cv2.VideoCapture(CFG['source_video']);prevs={};idx=0
 while targets:
  ok,img=cap.read()
  if not ok:break
  if idx in targets:
   fid=targets.pop(idx)
   for x1,y1,x2,y2 in CFG['graphics_masks_1280']:img[y1:y2,x1:x2]=48
   p=dest/(fid+'_previous.png');cv2.imwrite(str(p),img);prevs[fid]={'file':str(p.relative_to(ROOT)),'decoded_index':idx,'actual_pts':pts[idx],'sha256':sha(p)}
  idx+=1
 cap.release();assert not targets;dump(dest/'previous_frame_manifest.json',prevs)
 rows=[]
 for f in frames:
  t0=time.perf_counter();curr=cv2.imread(str(ROOT/f['file']));prev=cv2.imread(str(ROOT/prevs[f['id']]['file']))
  if a.mirror:curr=cv2.flip(curr,1);prev=cv2.flip(prev,1)
  cg=cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY);pg=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
  valid=np.full(cg.shape,255,np.uint8)
  for x1,y1,x2,y2 in CFG['graphics_masks_1280']:valid[y1:y2,x1:x2]=0
  valid[:8]=0;valid[-8:]=0;valid[:,:8]=0;valid[:,-8:]=0
  if a.mirror:valid=cv2.flip(valid,1)
  bg=valid.copy();fg=np.zeros_like(valid)
  boxes=json.loads(json.dumps([d for d in det[f['id']]['detections'] if d['score']>=.5]))
  if a.mirror:
   for d in boxes:
    l,t,r,b=d['box'];d['box']=[1279-r,t,1279-l,b]
  for d in boxes:
   l,t,r,b=map(int,d['box']);l=max(0,l-10);r=min(1280,r+10);t=max(0,t-10);b=min(720,b+10);bg[t:b,l:r]=0
   if d['class']=='horse':fg[t:b,l:r]=255
  fg=cv2.bitwise_and(fg,valid);bp,bq=tracked(cg,pg,bg,500);rec={'frame':f['id'],'pts':f['actual_pts'],'previous_pts':prevs[f['id']]['actual_pts'],'image_sha256':sha(ROOT/f['file']),'background_tracks':len(bp),'status':'unknown','direction_vector':None}
  if len(bp)>=20:
   if a.motion_model=='homography':H,inliers=cv2.findHomography(bp,bq,cv2.RANSAC,3.0)
   else:
    affine,inliers=cv2.estimateAffinePartial2D(bp,bq,method=cv2.RANSAC,ransacReprojThreshold=3.0)
    H=np.vstack([affine,[0.,0.,1.]]) if affine is not None else None
   ni=int(inliers.sum()) if inliers is not None else 0;rec['background_inliers']=ni
   if H is not None and ni>=20:
    pp,pq=tracked(cg,pg,fg,800);pred=cv2.perspectiveTransform(pp.reshape(-1,1,2),H).reshape(-1,2) if len(pp) else np.empty((0,2));res=pq-pred;mag=np.linalg.norm(res,axis=1);keep=np.isfinite(res).all(axis=1)&(mag>=1.0);res=res[keep];q=pq[keep];rec['foreground_tracks_after_background_compensation']=len(res);rec['background_H']=H.tolist();rec['residual_vectors']=res.tolist();rec['current_feature_points']=q.tolist()
    if len(res)>=10:
     unit=res/np.linalg.norm(res,axis=1)[:,None];concentration=float(np.linalg.norm(unit.mean(axis=0)));med=np.median(res,axis=0);rec['direction_concentration']=concentration;rec['median_residual_px']=med.tolist()
     if concentration>=.5 and np.linalg.norm(med)>0:
      rec['status']='ok';rec['direction_vector']=(med/np.linalg.norm(med)).tolist()
    im=Image.fromarray(cv2.cvtColor(curr,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(im);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    for point,vec in zip(q[::3],res[::3]):
     end=point+vec*3;draw.line([tuple(point),tuple(end)],fill='#ffe34d',width=2)
    if rec['direction_vector'] is not None:
     axis=np.array(rec['direction_vector']);arrow_start=np.array([500.,100.]);arrow_end=arrow_start+axis*100;perp=np.array([-axis[1],axis[0]]);draw.line([tuple(arrow_start),tuple(arrow_end)],fill='#40ff8a',width=5);draw.polygon([tuple(arrow_end),tuple(arrow_end-axis*16+perp*9),tuple(arrow_end-axis*16-perp*9)],fill='#40ff8a')
    draw.text((250,20),f'{f["id"]} flow={rec["status"]}; background inliers={ni}',font=font,fill='white',stroke_fill='black',stroke_width=2);im.save(dest/(f['id']+'_overlay.jpg'),quality=95)
  rec['elapsed_processing_ms']=(time.perf_counter()-t0)*1000;rows.append(rec);status('flow_direction','running',frame=f['id'],result=rec['status'])
 dump(result,rows);status('flow_direction','complete',tag=a.tag,frames=len(rows),usable=sum(r['status']=='ok' for r in rows))
if __name__=='__main__':main()
