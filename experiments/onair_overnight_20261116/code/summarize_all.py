"""Artifact-based summaries. Reference agreement is never called expert accuracy."""
import csv,json,math,statistics
from collections import Counter
from common import ROOT,dump,sha
DIR={'left':[-1,0],'down_left':[-1,1],'down':[0,1],'down_right':[1,1],'right':[1,0],'up_right':[1,-1],'up':[0,-1],'up_left':[-1,-1]}
def readlines(p):return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
def latency(rs,key='http_elapsed_ms'):
 x=sorted(r[key] for r in rs if key in r)
 def q(v):
  if not x:return None
  pos=(len(x)-1)*v;lo=int(pos);hi=min(lo+1,len(x)-1);return x[lo]*(hi-pos)+x[hi]*(pos-lo) if hi!=lo else x[lo]
 return {'n':len(x),'median_ms':statistics.median(x) if x else None,'p95_ms':q(.95)}
def angle(a,b):return math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(a,b))/(math.hypot(*a)*math.hypot(*b))))))
def main():
 ann=json.loads((ROOT/'geometry_annotations.json').read_text());core={f['id']:f for f in ann['frames']};vlm=readlines(ROOT/'raw/vlm.jsonl');lookup={r['item_id']:r for r in vlm};caps=readlines(ROOT/'raw/modular_cap.jsonl');tight=readlines(ROOT/'raw/manual_cap.jsonl');ledger=readlines(ROOT/'raw/request_ledger.jsonl');allcalls=vlm+caps+tight
 out={'reference_policy':'All new references are assistant visual drafts. Counts are agreement/coverage, not expert accuracy or independent-race performance.','model_request_count':len(ledger),'raw_responses':len(allcalls),'request_count_by_model':dict(Counter(x['model'] for x in ledger)),'http_failures':sum(r['failure'] for r in allcalls),'model_retries':sum(r.get('attempt',1)>1 for r in allcalls),'cpu_failures_recorded':len(readlines(ROOT/'evidence/execution_failures.jsonl')),'annotated_frames':len(core),'annotated_noses':sum(len(f['noses']) for f in core.values()),'excluded_noses':sum(len(f['excluded_nose_annotations']) for f in core.values()),'geometry_totals':json.loads((ROOT/'results/geometry_totals.json').read_text()),'total_http_seconds':sum(r['http_elapsed_ms'] for r in allcalls)/1000,'localization':[],'latencies':{},'view_diagnostics':{},'flow':{}}
 locrows=[]
 for suffix,label in [('_localize','9B_full_frame'),('_conditioned','9B_with_manual_view'),('_localize_27b','27B_full_frame')]:
  rows=[lookup[fid+suffix] for fid in core]
  for r in rows:
   f=core[r['frame_id']];p=r.get('parsed',{});gold=next(n for n in f['noses'] if n['reference_number']==f['leader_reference_number']);c=next(c for c in f['caps'] if c['id']==gold['reference_cap_id'])['center'];b=p.get('cap_box');nose=p.get('nose_xy');bpx=[b[0]*1.28,b[1]*.72,b[2]*1.28,b[3]*.72] if b else None;npx=[nose[0]*1.28,nose[1]*.72] if nose else None
   locrows.append({'frame':f['id'],'condition':label,'leader_claimed':p.get('leader_visible',False),'cap_color':p.get('cap_color'),'cap_center_covered_by_predicted_box':bool(bpx and bpx[0]<=c[0]<=bpx[2] and bpx[1]<=c[1]<=bpx[3]),'cap_color_matches_reference_purple':p.get('cap_color')=='purple','nose_distance_to_reference_px':math.dist(npx,gold['point']) if npx else None,'http_ms':r['http_elapsed_ms'],'reference_status':'assistant provisional'})
  rr=[x for x in locrows if x['condition']==label];out['localization'].append({'condition':label,'n':len(rr),'leader_claims':sum(x['leader_claimed'] for x in rr),'leader_abstain':sum(not x['leader_claimed'] for x in rr),'cap_box_reference_center_coverage':sum(x['cap_center_covered_by_predicted_box'] for x in rr),'purple_class_reference_agreement':sum(x['cap_color_matches_reference_purple'] for x in rr)})
  out['latencies'][label]=latency(rows)
 with (ROOT/'results/localization_comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(locrows[0]));w.writeheader();w.writerows(locrows)
 for suffix,label in [('_view','9B_core_view'),('_view_27b','27B_core_view')]:
  rows=[lookup[fid+suffix] for fid in core];out['view_diagnostics'][label]={'n':len(rows),'predicted_views':dict(Counter(r['parsed']['view'] for r in rows)),'predicted_directions':dict(Counter(r['parsed']['direction'] for r in rows))};out['latencies'][label]=latency(rows)
 eq=[]
 for fid in core:
  p=lookup[fid+'_view']['parsed'];m=lookup[fid+'_view_mirror']['parsed'];a=DIR.get(p['direction']);b=DIR.get(m['direction']);eq.append({'frame':fid,'original':p['direction'],'mirror':m['direction'],'reflection_consistent':bool(a and b and [-a[0],a[1]]==b),'same_view_class':p['view']==m['view']})
 out['view_diagnostics']['9B_mirror']={'consistent':sum(x['reflection_consistent'] for x in eq),'n':len(eq),'rows':eq}
 out['view_diagnostics']['enum_order_control']={'n':len(core),'same_parsed_response':sum(lookup[fid+'_view']['parsed']==lookup[fid+'_view_front_first']['parsed'] for fid in core),'interpretation':'This enum reorder did not change these responses; does not exclude other prompt/schema effects.'}
 out['modular']={}
 for mode in ['manual_axis','model9b_axis']:
  rows=[x for x in json.loads((ROOT/'results/modular_pipeline.json').read_text()) if x['axis_source']==mode]
  out['modular'][mode]={'n':len(rows),'selected_horse_reference_nose_coverage':sum(x.get('selected_horse_contains_reference_leader_nose',False) for x in rows),'roi_reference_cap_center_coverage':sum(x.get('roi_contains_reference_leader_cap_center',False) for x in rows),'purple_class_reference_agreement':sum(x.get('number_matches_reference_leader',False) for x in rows),'classification_unknown':sum(x.get('predicted_number')==0 for x in rows),'nonpurple_nonunknown':sum(x.get('predicted_number') not in [None,0,9] for x in rows)}
 out['tight_cap']={'n':len(tight),'purple':sum(r.get('number')==9 for r in tight),'unknown':sum(r.get('number')==0 for r in tight),'reference_status':'same8frames and samepurplehorse, posthoc manually tight crops, not 11classaccuracy'};out['latencies']['tight_cap']=latency(tight);out['latencies']['modular_cap_unique_calls']=latency(caps)
 for tag in ['flow','flow_01_homography','flow_05_affine','flow_01_affine']:
  rows=json.loads((ROOT/'results'/(tag+'_direction.json')).read_text());vals=[]
  for r in rows:vals.append({'frame':r['frame'],'angle_to_manual_heuristic_degrees':angle(r['direction_vector'],core[r['frame']]['direction']) if r['direction_vector'] else None,'status':r['status']})
  out['flow'][tag]={'n':len(rows),'usable':sum(r['status']=='ok' for r in rows),'agreement_with_heuristic_within45deg':sum(x['angle_to_manual_heuristic_degrees'] is not None and x['angle_to_manual_heuristic_degrees']<=45 for x in vals),'NOT_ACCURACY':'Manual axes are heuristic and uncalibrated; angle differences do not establish incorrect motion.','details':vals,'timing':latency(rows,'elapsed_processing_ms')}
 orig={x['frame']:x for x in json.loads((ROOT/'results/flow_01_affine_direction.json').read_text())};mirror=json.loads((ROOT/'results/flow_01_affine_mirror_direction.json').read_text());eq=[]
 for r in mirror:
  a=orig[r['frame']]['direction_vector'];b=r['direction_vector'];deviation=angle([-a[0],a[1]],b) if a and b else None;eq.append({'frame':r['frame'],'angle_deviation_degrees':deviation,'within15deg':deviation is not None and deviation<=15})
 out['flow_mirror']={'consistent_within15deg':sum(x['within15deg'] for x in eq),'n':len(eq),'details':eq,'limitation':'Geometric equivariance check, not proof of correct real-world motion or ranking.'}
 out['rank_projection']=json.loads((ROOT/'results/rank_projection_summary.json').read_text())
 detections=readlines(ROOT/'raw/detector.jsonl');out['detector']={'frames':len(detections),'model':'torchvision FasterRCNN ResNet50 FPNv2 COCO_V1','training_steps':0,'nose_or_helmet_detector':False,'all_frame_forward_timing':latency(detections,'elapsed_forward_ms'),'note':'Includes cold forward calls; excludes image decode and host-to-device tensor preparation. Detection count is not precision/recall.'}
 dump(ROOT/'results/SUMMARY.json',out);print(json.dumps({k:out[k] for k in ['model_request_count','request_count_by_model','http_failures','localization','modular','tight_cap','latencies','rank_projection']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
