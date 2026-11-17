"""Frozen new-frame association diagnostic, with declared provisional references."""
import csv,json,time
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import ROOT,sha,dump,append,status

METHODS=['A_distance','B_view_gate','C_view_gate_unique','D_view_gate_abstain','E_cut_only_gate']
def solve(N,C,D,method):
 diff=C[None,:,:]-N[:,None,:];cost=np.linalg.norm(diff,axis=2)
 if method!='A_distance':cost[(diff@(D/np.linalg.norm(D)))>0]=np.inf
 if method=='C_view_gate_unique':
  rows,cols=linear_sum_assignment(np.where(np.isfinite(cost),cost,1e8));pred=np.full(len(N),-1,int);pred[rows]=cols
  pred[[i for i,j in enumerate(pred) if not np.isfinite(cost[i,j])]]=-1
 else:
  pred=cost.argmin(axis=1);pred[np.isinf(cost.min(axis=1))]=-1
 if method=='D_view_gate_abstain':
  for i,j in enumerate(pred):
   if j<0:continue
   ordered=np.sort(cost[i]);margin=(ordered[1]-ordered[0])/max(ordered[0],1)
   if margin<.10:pred[i]=-1
 return pred,cost

def main():
 ann=json.loads((ROOT/'geometry_annotations.json').read_text());lock=json.loads((ROOT/'evidence/annotation_lock.json').read_text());assert sha(ROOT/'geometry_annotations.json')==lock['annotations_sha256']
 rng=np.random.default_rng(20261116);rows=[];summary=[];raw=[];ts=[]
 for f in ann['frames']:
  caps=f['caps'];noses=f['noses'];cp=rng.permutation(len(caps));np_=rng.permutation(len(noses));caps=[caps[i] for i in cp];noses=[noses[i] for i in np_]
  C=np.array([c['center'] for c in caps],float);N=np.array([n['point'] for n in noses],float);D=np.array(f['direction'],float)
  target=np.array([next(i for i,c in enumerate(caps) if c['id']==n['reference_cap_id']) for n in noses])
  # Additional deliberately frozen-per-shot diagnostic. First annotated direction
  # in each shot is retained, not re-estimated while horses turn within the shot.
  shotD=np.array([1,.4] if f['pts']<32.733333 else ([-1,.4] if f['pts']<43.066667 else [-.2,1]),float)
  for method in METHODS:
   axis=shotD if method=='E_cut_only_gate' else D
   pred,cost=solve(N,C,axis,method);samples=[]
   for _ in range(20):solve(N,C,axis,method)
   for _ in range(1000):
    t=time.perf_counter_ns();solve(N,C,axis,method);samples.append((time.perf_counter_ns()-t)/1e6)
   jitter_good=np.zeros(len(N),int);jitter_wrong=np.zeros(len(N),int);jitter_abstain=np.zeros(len(N),int)
   for _ in range(2000):
    jp,_=solve(N+rng.uniform(-5,5,N.shape),C+rng.uniform(-2,2,C.shape),axis,method)
    jitter_good+=(jp==target);jitter_wrong+=((jp!=target)&(jp>=0));jitter_abstain+=(jp<0)
   for i,n in enumerate(noses):
    chosen=pred[i];valid=np.sort(cost[i][np.isfinite(cost[i])]);gap=float((valid[1]-valid[0])/max(valid[0],1)) if len(valid)>1 else None
    rows.append({'frame':f['id'],'pts':f['pts'],'view':f['view'],'nose_id':n['id'],'reference_number':n['reference_number'],'method':method,'predicted_number':caps[chosen]['display_number'] if chosen>=0 else None,'agreement':bool(chosen==target[i]),'abstain':bool(chosen<0),'wrong':bool(chosen>=0 and chosen!=target[i]),'top_two_relative_gap':gap,'jitter_agreement_fraction':float(jitter_good[i]/2000),'jitter_wrong_fraction':float(jitter_wrong[i]/2000),'jitter_abstain_fraction':float(jitter_abstain[i]/2000),'reference_status':'assistant visual draft, not human/expert verified'})
   summary.append({'frame':f['id'],'method':method,'n':len(N),'agreement':int(np.sum(pred==target)),'wrong':int(np.sum((pred>=0)&(pred!=target))),'abstain':int(np.sum(pred<0))})
   ts.append({'frame':f['id'],'method':method,'median_ms':float(np.median(samples)),'p95_ms':float(np.percentile(samples,95)),'scope':'only coordinate association, not detection/recognition/annotation'})
   raw.append({'frame':f['id'],'method':method,'direction':axis.tolist(),'nose_coordinates':N.tolist(),'cap_coordinates':C.tolist(),'nose_ids_outside_solver':[n['id'] for n in noses],'cap_ids_outside_solver':[c['id'] for c in caps],'predicted_indices':pred.tolist(),'cost_matrix':[[None if not np.isfinite(v) else float(v) for v in row] for row in cost],'timing_ms':samples})
 with (ROOT/'results/geometry_comparison.csv').open('w',newline='') as g:
  w=csv.DictWriter(g,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump(ROOT/'results/geometry_summary.json',summary);dump(ROOT/'results/geometry_timing.json',ts)
 (ROOT/'raw/geometry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
 totals={m:{k:sum(s[k] for s in summary if s['method']==m) for k in ['n','agreement','wrong','abstain']} for m in METHODS}
 dump(ROOT/'results/geometry_totals.json',totals)
 status('geometry','complete',totals=totals,annotation_sha256=lock['annotations_sha256'])
if __name__=='__main__':main()
