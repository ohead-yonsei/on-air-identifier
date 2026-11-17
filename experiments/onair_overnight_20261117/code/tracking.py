"""Causal geometric horse/person tracking. IDs are not horse numbers."""
import argparse,collections,json,math,time
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from common import *
from regions import area,iou
from archive.flow_direction import tracked

def associate(ds):
 hs=[d for d in ds if d['class']=='horse'];ps=[d for d in ds if d['class']=='person'];cost=np.full((len(hs),len(ps)+len(hs)),1.)
 for i,h in enumerate(hs):
  l,t,r,b=h['box'];w=r-l;hh=b-t;cx=(l+r)/2
  for j,p in enumerate(ps):
   pl,pt,pr,pb=p['box'];px=(pl+pr)/2;py=(pt+pb)/2
   cost[i,j]=math.hypot((px-cx)/w,(py-(t+.1*hh))/hh)-.05*p['score'] if l-.2*w<=px<=r+.2*w and t-.75*hh<=py<=b-.2*hh else 10
 if not hs:return [],[p['id'] for p in ps]
 ri,cj=linear_sum_assignment(cost);pairs=[]
 for i,j in zip(ri,cj):pairs.append({'horse':hs[i]['id'],'person':ps[j]['id'] if j<len(ps) and cost[i,j]<1 else None,'cost':float(cost[i,j])})
 used={p['person'] for p in pairs if p['person'] is not None};return pairs,[p['id'] for p in ps if p['id'] not in used]

def affine_flow(prev,curr,ds):
 start=time.perf_counter();pg=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY);cg=cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY);valid=np.full(cg.shape,255,np.uint8);valid[:8]=0;valid[-8:]=0;valid[:,:8]=0;valid[:,-8:]=0
 valid[:110,:240]=0;valid[610:]=0;bg=valid.copy();fg=np.zeros_like(bg)
 for d in ds:
  l,t,r,b=d['box'];l=max(0,int(l)-10);r=min(cg.shape[1],int(r)+10);t=max(0,int(t)-10);b=min(cg.shape[0],int(b)+10);bg[t:b,l:r]=0
  if d['class']=='horse':fg[t:b,l:r]=255
 fg=cv2.bitwise_and(fg,valid);bp,bq=tracked(cg,pg,bg,500);rec={'background_tracks':len(bp),'inliers':0,'affine':None,'direction':None,'direction_status':'unknown'}
 if len(bp)>=20:
  H,ins=cv2.estimateAffinePartial2D(bp,bq,method=cv2.RANSAC,ransacReprojThreshold=3.0);ni=int(ins.sum()) if ins is not None else 0;rec['inliers']=ni
  if H is not None and ni>=20:
   rec['affine']=H.tolist();pp,pq=tracked(cg,pg,fg,800)
   if len(pp):
    pred=cv2.transform(pp.reshape(-1,1,2),H).reshape(-1,2);res=pq-pred;mag=np.linalg.norm(res,axis=1);res=res[np.isfinite(res).all(axis=1)&(mag>=1.)];rec['foreground_tracks']=len(res)
    if len(res)>=10:
     conc=float(np.linalg.norm((res/np.linalg.norm(res,axis=1)[:,None]).mean(axis=0)));med=np.median(res,axis=0);rec['concentration']=conc;rec['median_residual']=med.tolist()
     if conc>=.5 and np.linalg.norm(med)>0:rec['direction']=(med/np.linalg.norm(med)).tolist();rec['direction_status']='usable_motion_only'
 rec['processing_ms']=(time.perf_counter()-start)*1000;return rec

def cut_metrics(prev,curr):
 p=cv2.resize(prev,(64,36));c=cv2.resize(curr,(64,36));diff=float(np.mean(np.abs(p.astype(float)-c)))
 def hist(im):
  h=cv2.calcHist([cv2.cvtColor(im,cv2.COLOR_BGR2HSV)],[0,1],None,[32,32],[0,180,0,256]);return cv2.normalize(h,h).flatten()
 corr=float(cv2.compareHist(hist(p),hist(c),cv2.HISTCMP_CORREL));return {'mean_rgb_abs_diff':diff,'hsv_hist_correlation':corr,'cut':diff>25 or corr<.65}

def warp_box(box,A):
 if A is None:return list(box)
 l,t,r,b=box;pts=cv2.transform(np.array([[[l,t],[r,t],[r,b],[l,b]]],np.float32),np.asarray(A,np.float32))[0];return [float(pts[:,0].min()),float(pts[:,1].min()),float(pts[:,0].max()),float(pts[:,1].max())]

class Tracker:
 def __init__(self,use_affine,dummy=1.1):self.use_affine=use_affine;self.dummy=dummy;self.active=[];self.next_id=1;self.shot=0
 def step(self,ds,flow,cut):
  if cut:self.active=[];self.shot+=1
  A=flow.get('affine') if self.use_affine else None
  for t in self.active:t['box']=warp_box(t['box'],A);t['missed']+=1
  cost=np.full((len(ds),len(self.active)+len(ds)),self.dummy)
  for i,d in enumerate(ds):
   for j,t in enumerate(self.active):
    if d['class']!=t['class']:cost[i,j]=10;continue
    a=d['box'];b=t['box'];overlap=iou(a,b);dist=math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2)/max(1,math.hypot(b[2]-b[0],b[3]-b[1]));cost[i,j]=(1-overlap)+.25*dist if overlap>=.1 or dist<=1 else 10
  assigned={};created=[]
  if ds:
   ri,cj=linear_sum_assignment(cost)
   for i,j in zip(ri,cj):
    if j<len(self.active) and cost[i,j]<self.dummy:
     tr=self.active[j];tr['box']=list(ds[i]['box']);tr['missed']=0
    else:
     tr={'track_id':self.next_id,'class':ds[i]['class'],'box':list(ds[i]['box']),'missed':0};self.next_id+=1;created.append(tr)
    assigned[ds[i]['id']]=tr['track_id']
  self.active=[t for t in self.active if t['missed']<=3]+created
  return {'detection_to_track':assigned,'new_tracks':len(created),'active_tracks':len(self.active),'shot':self.shot}

def reference_metrics(rows):
 ann=json.loads((Path(CFG['old_experiment'])/'geometry_annotations.json').read_text());bytime={round(r['pts'],1):r for r in rows};out=[]
 for f in ann['frames']:
  r=bytime[round(f['actual_pts'],1)] if 'actual_pts' in f else bytime[round(f['pts'],1)]
  for n in f['noses']:
   cap=next(c['center'] for c in f['caps'] if c['id']==n['reference_cap_id']);nose=n['point'];hs=[d for d in r['detections'] if d['class']=='horse'];ps=[d for d in r['detections'] if d['class']=='person']
   def containing(point,boxes):
    choices=[]
    for d in boxes:
     l,t,rr,b=d['box'];w=rr-l;h=b-t
     if l<=point[0]<=rr and t<=point[1]<=b:choices.append((area(d['box']),d))
    return min(choices,key=lambda x:x[0])[1] if choices else None
   h=containing(nose,hs);p=containing(cap,ps);paired=next((x['person'] for x in r['horse_person_pairs'] if h and x['horse']==h['id']),None)
   item={'frame':f['id'],'pts':r['pts'],'number_draft':n['reference_number'],'nose_horse_detection':h['id'] if h else None,'cap_person_detection':p['id'] if p else None,'paired_person':paired,'association_agrees_draft':bool(h and p and paired==p['id']),'tracks':{}}
   for name,tr in r['trackers'].items():item['tracks'][name]={'shot':tr['shot'],'horse_track':tr['detection_to_track'].get(h['id']) if h else None,'person_track':tr['detection_to_track'].get(p['id']) if p else None}
   out.append(item)
 summaries={}
 for mode in rows[0]['trackers']:
  validpairs=changes=across_cut=missing=0;last={}
  for row in sorted(out,key=lambda x:x['pts']):
   n=row['number_draft'];tr=row['tracks'][mode]
   if tr['person_track'] is None:missing+=1;continue
   if n in last:
    old=last[n]
    if old['shot']==tr['shot']:validpairs+=1;changes+=int(old['person_track']!=tr['person_track'])
    else:across_cut+=1
   last[n]=tr
  summaries[mode]={'sparse_same_shot_transitions':validpairs,'sparse_person_id_changes':changes,'cross_cut_transitions_excluded':across_cut,'missing_reference_person':missing,'new_tracks_over_sequence':sum(r['trackers'][mode]['new_tracks'] for r in rows),'not_MOT_ID_switch_ground_truth':True}
 return {'references':out,'summary':summaries,'association_draft_matches':sum(x['association_agrees_draft'] for x in out),'reference_count':len(out),'reference_policy':'Assistant draft 22 nose/cap points, not human MOT identities. Smallest containing box is an evaluation proxy, not true instance segmentation.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--extras',action='store_true');a=ap.parse_args();cv2.setNumThreads(4);fm=json.loads((ROOT/'tracking_frame_manifest.json').read_text());det={r['frame']:r for r in readlines(ROOT/'raw/tracking_detector.jsonl')};out=ROOT/'raw/tracking.jsonl';flow_cache={r['frame']:r for r in readlines(ROOT/'raw/tracking_flow.jsonl')};modes={'no_warp':Tracker(False),'affine':Tracker(True)}
 if a.extras:modes.update({'affine_dummy08':Tracker(True,.8),'affine_dummy12':Tracker(True,1.2)})
 rows=[];prev=None
 for f in fm:
  guard();curr=cv2.imread(str(ROOT/f['file']));ds=det[f['id']]['detections'];pairs,unassigned=associate(ds)
  if f['id'] in flow_cache:fr=flow_cache[f['id']]
  else:
   cut={'cut':True,'reason':'first_frame'} if prev is None else cut_metrics(prev,curr)
   flow={'affine':None,'direction':None,'direction_status':'cut_or_initial'} if cut['cut'] else affine_flow(prev,curr,ds)
   ratios=[(d['box'][2]-d['box'][0])/max(1,d['box'][3]-d['box'][1]) for d in ds if d['class']=='horse'];aspect=float(np.median(ratios)) if ratios else None;view='unknown' if aspect is None else 'side_like' if aspect>=1.6 else 'front_or_rear_like' if aspect<=1.1 else 'oblique_like'
   fr={'frame':f['id'],'pts':f['actual_pts'],'previous_pts':rows[-1]['pts'] if rows else None,'cut_metrics':cut,'flow':flow,'view_proxy':view,'horse_box_median_aspect':aspect,'view_policy':'Updated every frame from horse-box aspect; unvalidated proxy, front vs rear unresolved; motion is not rank calibration'};append(ROOT/'raw/tracking_flow.jsonl',fr)
  row={**fr,'detections':ds,'horse_person_pairs':pairs,'unassigned_people':unassigned,'trackers':{name:tr.step(ds,fr['flow'],fr['cut_metrics']['cut']) for name,tr in modes.items()}};rows.append(row);prev=curr
  if len(rows)%30==0:status('E_tracking','running',frames=len(rows),extras=a.extras)
 dest=ROOT/'raw'/('tracking_extra.jsonl' if a.extras else 'tracking.jsonl')
 assert not dest.exists(),'preserve existing tracking result'
 for row in rows:append(dest,row)
 summary=reference_metrics(rows);summary.update(frames=len(rows),cuts=[r['pts'] for r in rows if r['cut_metrics']['cut']],view_proxy_counts=dict(collections.Counter(r['view_proxy'] for r in rows)),horse_person_pairs=sum(sum(x['person'] is not None for x in r['horse_person_pairs']) for r in rows),unassigned_horses=sum(sum(x['person'] is None for x in r['horse_person_pairs']) for r in rows),unassigned_people=sum(len(r['unassigned_people']) for r in rows))
 dump(ROOT/'results'/('tracking_extra_summary.json' if a.extras else 'tracking_summary.json'),summary);status('E_tracking','complete',extras=a.extras,frames=len(rows),draft_matches=summary['association_draft_matches'],references=summary['reference_count'])
if __name__=='__main__':main()
