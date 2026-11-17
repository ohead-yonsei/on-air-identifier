"""Fixed region generation; reference boxes used only for U and scoring."""
import math,time
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1]) if b is not None else 0.
def inter(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1])) if a is not None and b is not None else 0.
def iou(a,b):return inter(a,b)/max(1e-9,area(a)+area(b)-inter(a,b))
def clip(b,w,h):
 if b is None:return None
 l,t,r,d=b;b=[max(0,math.floor(l)),max(0,math.floor(t)),min(w,math.ceil(r)),min(h,math.ceil(d))]
 return b if area(b)>0 else None
def pose_match(people,poses):
 out={}
 if not people or not poses:return out
 costs=np.ones((len(people),len(poses)+len(people)))*.7
 for i,p in enumerate(people):
  for j,q in enumerate(poses):costs[i,j]=1-iou(p['box'],q['box']) if iou(p['box'],q['box'])>=.3 else 10
 ri,cj=linear_sum_assignment(costs)
 for i,j in zip(ri,cj):
  if j<len(poses) and costs[i,j]<=.7:out[people[i]['id']]=poses[j]
 return out
def region(im,b,method,pose=None,manual=None):
 start=time.perf_counter();h,w=im.shape[:2];info={};box=None
 if method=='U_manual':box=manual
 elif b is None:info['reason']='no_common_person_detection'
 else:
  l,t,r,d=b;bw=r-l;bh=d-t
  if method=='R0_27':box=[l+.08*bw,t,r-.08*bw,t+max(4,.27*bh)]
  elif method=='R0_35':box=[l,t,r,t+.35*bh]
  elif method=='R1_pose':
   if pose is None:info['reason']='no_pose_match'
   else:
    pts=np.asarray(pose['keypoints'])[:5,:2];sc=np.asarray(pose['keypoints_scores'])[:5];good=pts[sc>=2];info.update(face_keypoints=pts.tolist(),keypoint_scores=sc.tolist(),usable_points=len(good))
    if len(good)<2:info['reason']='insufficient_face_keypoints'
    else:
     cx,cy=np.median(good,axis=0);cy-=.07*bh;rw=float(np.clip(max(.25*bw,1.5*np.ptp(good[:,0])),.2*bw,.8*bw));rh=.85*rw;box=[cx-rw/2,cy-rh/2,cx+rw/2,cy+rh/2]
  elif method=='R2_cluster':
   base=clip([l,t,r,t+.35*bh],w,h);x1,y1,x2,y2=base;crop=im[y1:y2,x1:x2];hh,ww=crop.shape[:2]
   if hh*ww<16:info['reason']='region_too_small'
   else:
    lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB).astype(np.float32);cv2.setRNGSeed(20261117);_,labels,centers=cv2.kmeans(lab.reshape(-1,3),4,None,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,30,.2),1,cv2.KMEANS_PP_CENTERS);labids=labels.reshape(hh,ww);border=np.median(np.concatenate([lab[0],lab[:,0],lab[:,-1]]),axis=0);candidates=[]
    for k in range(4):
     n,cc,stats,centroids=cv2.connectedComponentsWithStats((labids==k).astype(np.uint8),8)
     for j in range(1,n):
      xx,yy,cw,ch,sz=stats[j];frac=float(sz/(hh*ww));cx,cy=centroids[j];bd=float(np.linalg.norm(centers[k]-border))
      if not .01<=frac<=.55 or bd<15:continue
      distance=math.hypot(cx/max(ww,1)-.5,cy/max(hh,1)-.25);score=distance+.3*abs(math.log(max(frac,1e-6)/.15))-.001*min(bd,100)
      candidates.append({'cost':score,'cluster':k,'component':j,'area_fraction':frac,'border_delta_lab':bd,'bbox':[int(x1+xx-1),int(y1+yy-1),int(x1+xx+cw+1),int(y1+yy+ch+1)]})
    info['candidates']=candidates
    if candidates:best=min(candidates,key=lambda x:x['cost']);box=best['bbox'];info['chosen']=best
    else:info['reason']='no_colour_component'
 box=clip(box,w,h)
 return box,info,(time.perf_counter()-start)*1000
