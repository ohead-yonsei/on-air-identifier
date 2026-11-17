"""Meaningful integrity checks on actual executed requests and frozen inputs."""
import base64,hashlib,json,csv,math
from pathlib import Path
from PIL import Image
from common import ROOT,CFG,sha,dump
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def main():
 checks=[]
 def check(name,ok):
  checks.append({'name':name,'passed':bool(ok)})
  if not ok:raise AssertionError(name)
 ledger=lines(ROOT/'raw/request_ledger.jsonl');allresponses=sum([lines(ROOT/'raw'/p) for p in ['vlm.jsonl','modular_cap.jsonl','manual_cap.jsonl']],[]);lookup={r['item_id']:r for r in allresponses}
 check('97actual_requests_and_responses',len(ledger)==len(allresponses)==97)
 check('unique_request_item_ids',len(lookup)==97==len(set(r['item_id'] for r in ledger)))
 check('within_declared_request_cap',len(ledger)<=CFG['max_vlm_requests_including_failures'])
 for r in ledger:
  image=ROOT/r['image'];check(r['item_id']+'_image_hash',sha(image)==r['image_sha256']);check(r['item_id']+'_no_review_overlay_input',Path(r['image']).parts[0] not in ['review','highlights'])
  messages=json.loads(json.dumps(r['messages']));messages[-1]['images']=[base64.b64encode(image.read_bytes()).decode()]
  payload={'model':r['model'],'messages':messages,'format':r['schema'],'stream':False,'think':False,'keep_alive':'10m','options':r['options']}
  serialized=json.dumps(payload,ensure_ascii=False).encode();check(r['item_id']+'_exact_request_reconstruction',hashlib.sha256(serialized).hexdigest()==r['request_sha256'])
  resp=lookup[r['item_id']];check(r['item_id']+'_successful_independent_call',not resp['failure'] and resp.get('http_status')==200 and len(messages)==2)
  check(r['item_id']+'_raw_response_retained',bool(resp.get('raw_http_response')))
 ann=json.loads((ROOT/'geometry_annotations.json').read_text());lock=json.loads((ROOT/'evidence/annotation_lock.json').read_text());check('annotations_unchanged_after_lock',sha(ROOT/'geometry_annotations.json')==lock['annotations_sha256']);check('8frames22scored2excluded',len(ann['frames'])==8 and sum(len(f['noses']) for f in ann['frames'])==22 and sum(len(f['excluded_nose_annotations']) for f in ann['frames'])==2)
 for f in ann['frames']:check(f['id']+'_frozen_source_hash',sha(ROOT/f['file'])==f['source_sha256'])
 manifest=json.loads((ROOT/'frame_manifest.json').read_text());pts=json.loads((ROOT/'evidence/video_pts.json').read_text())
 for f in manifest:check(f['file']+'_exact_pts_and_hash',f['actual_pts']==pts[f['decoded_index']] and f['actual_pts']<=f['target_seconds']+1e-8 and sha(ROOT/f['file'])==f['sha256'])
 for tag in ['flow','flow_01_homography','flow_05_affine','flow_01_affine','flow_01_affine_mirror']:
  prev=json.loads((ROOT/tag/'previous_frame_manifest.json').read_text());cfg=json.loads((ROOT/tag/'config.json').read_text());rows=json.loads((ROOT/'results'/(tag+'_direction.json')).read_text());check(tag+'_8complete',len(rows)==8)
  for r in rows:
   p=prev[r['frame']];check(tag+r['frame']+'_causal_pts_and_hash',p['actual_pts']==pts[p['decoded_index']] and p['actual_pts']<=r['pts']-cfg['lookback_seconds']+1e-8 and sha(ROOT/p['file'])==p['sha256'])
 geometry=list(csv.DictReader((ROOT/'results/geometry_comparison.csv').open()));check('110geometry_decisions',len(geometry)==110)
 for m in set(r['method'] for r in geometry):check(m+'_22exclusive_outcomes',sum(r['method']==m for r in geometry)==22 and all(sum(r[k]=='True' for k in ['agreement','wrong','abstain'])==1 for r in geometry if r['method']==m))
 pairs=json.loads((ROOT/'results/modular_pipeline.json').read_text());check('16modular_cases14unique_calls',len(pairs)==16 and len(lines(ROOT/'raw/modular_cap.jsonl'))==14)
 protocol=json.loads((ROOT/'protocols/cap_classifier.json').read_text())
 for r in lines(ROOT/'raw/modular_cap.jsonl')+lines(ROOT/'raw/manual_cap.jsonl'):check(r['item_id']+'_identical_cap_protocol',r['messages'][0]['content']==protocol['SYSTEM'] and r['messages'][1]['content']==protocol['PROMPT'] and r['schema']==protocol['SCHEMA'] and r['options']==protocol['options'])
 review=list(csv.DictReader((ROOT/'review/USER_REVIEW.csv').open()));check('no_fabricated_human_review',len(review)==22 and all(not r['user_confirmed'] for r in review))
 dump(ROOT/'evidence/verification_final.json',{'checks_passed':len(checks),'checks':checks,'meaning':'request/pixel/timestamp/accounting integrity, not model accuracy validation'});print(json.dumps({'checks_passed':len(checks)},indent=2))
if __name__=='__main__':main()
