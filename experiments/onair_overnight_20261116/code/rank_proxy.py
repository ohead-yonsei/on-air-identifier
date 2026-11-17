"""Projection diagnostics; no claim that pixel projection is calibrated race rank."""
import csv,json,math
from common import ROOT,dump,status
DIR={'left':[-1,0],'down_left':[-1,1],'down':[0,1],'down_right':[1,1],'right':[1,0],'up_right':[1,-1],'up':[0,-1],'up_left':[-1,-1]}
def main():
 ann=json.loads((ROOT/'geometry_annotations.json').read_text());vlm={r['item_id']:r for r in map(json.loads,(ROOT/'raw/vlm.jsonl').read_text().splitlines())};det={r['id']:r for r in map(json.loads,(ROOT/'raw/detector.jsonl').read_text().splitlines())}
 flow={tag:{r['frame']:r for r in json.loads((ROOT/'results'/(tag+'_direction.json')).read_text())} for tag in ['flow','flow_01_homography','flow_05_affine','flow_01_affine']};rows=[]
 for f in ann['frames']:
  axes={'manual_heuristic':f['direction'],'vlm9':DIR.get(vlm[f['id']+'_view']['parsed']['direction']),'vlm27':DIR.get(vlm[f['id']+'_view_27b']['parsed']['direction'])}
  axes.update({tag:fr[f['id']]['direction_vector'] for tag,fr in flow.items()})
  gold=next(n for n in f['noses'] if n['reference_number']==f['leader_reference_number'])
  for name,d in axes.items():
   row={'frame':f['id'],'axis_source':name,'axis':d,'reference_status':'assistant visual draft; no calibrated rank truth','abstain':d is None}
   if d is not None:
    norm=math.hypot(*d);axis=[q/norm for q in d]
    selected=max(f['noses'],key=lambda n:sum(x*y for x,y in zip(n['point'],axis)))
    h=[x for x in det[f['id']]['detections'] if x['class']=='horse' and x['score']>=.5]
    box=max(h,key=lambda x:((x['box'][0]+x['box'][2])/2)*axis[0]+((x['box'][1]+x['box'][3])/2)*axis[1])['box'] if h else None
    n=gold['point'];row.update(selected_nose_reference_number=selected['reference_number'],nose_choice_agrees_with_visual_draft=selected['reference_number']==f['leader_reference_number'],selected_automatic_horse_box=box,automatic_box_contains_reference_leader_nose=bool(box and box[0]<=n[0]<=box[2] and box[1]<=n[1]<=box[3]))
   rows.append(row)
 dump(ROOT/'results/rank_projection.json',rows)
 summary=[]
 for name in sorted(set(r['axis_source'] for r in rows)):
  rs=[r for r in rows if r['axis_source']==name];summary.append({'axis_source':name,'n':len(rs),'abstain':sum(r['abstain'] for r in rs),'nose_reference_agreement':sum(r.get('nose_choice_agrees_with_visual_draft',False) for r in rs),'auto_box_reference_coverage':sum(r.get('automatic_box_contains_reference_leader_nose',False) for r in rs)})
 dump(ROOT/'results/rank_projection_summary.json',summary);status('rank_projection','complete',summary=summary)
if __name__=='__main__':main()
