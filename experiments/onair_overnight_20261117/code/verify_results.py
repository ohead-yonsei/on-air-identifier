"""Read-only checks of saved inputs/results; writes only a verification record."""
import base64,csv,hashlib,json,re,statistics
from pathlib import Path
from PIL import Image
import numpy as np
from common import ROOT,CFG,sha,dump,readlines,now,gpu_check

def main():
 checks=[]
 def check(name,ok,**details):
  checks.append({'check':name,'passed':bool(ok),**details})
 # Frozen experimental definitions were not edited after evaluation.
 for name in ['plan_lock.json','evaluation_lock.json']:
  lock=json.loads((ROOT/'evidence'/name).read_text())
  for path,digest in lock['files'].items():check(name+':'+path,sha(ROOT/path)==digest)
 ann=json.loads((ROOT/'annotations.json').read_text())['samples'];lock=json.loads((ROOT/'evidence/annotation_lock.json').read_text());check('annotation_frozen',sha(ROOT/'annotations.json')==lock['sha256'])
 check('labels_scope',len(ann)==66 and sum(x['split']=='human_check' for x in ann)==6 and sum(not x['number'] for x in ann)==4)
 hl=json.loads((ROOT/'evidence/human_labels_1010.json').read_text())['labels'];check('human_colours_only_preserved',all(s['number']==hl[s['id'].split('_')[-1]] for s in ann if s['split']=='human_check'))
 check('old_human_labels_unchanged',sha(Path(CFG['old_cap_experiment'])/'human_labels.json')==sha(ROOT/'evidence/human_labels_1010.json'))
 # Exact original source bytes and extracted source-frame manifests.
 sources=json.loads((ROOT/'evidence/video_sources_original.json').read_text())['videos']
 for s in sources:check('original_video:'+s['race'],sha(ROOT/s['file'])==s['sha256'])
 oldcfg=json.loads((Path(CFG['old_experiment'])/'config.json').read_text());check('original_october_video',sha(CFG['old_video'])==oldcfg['source_sha256'])
 manifests=[]
 for name,n in [('frame_manifest.json',21),('tracking_frame_manifest.json',271)]:
  fs=json.loads((ROOT/name).read_text());manifests+=fs;check(name+':count',len(fs)==n)
  check(name+':past_only',all(f['actual_pts']<=f.get('target_pts',f['actual_pts'])+1e-8 for f in fs))
  check(name+':image_hashes',all(sha(ROOT/f['file'])==f['sha256'] for f in fs))
  check(name+':dimensions',all(Image.open(ROOT/f['file']).size==(f['width'],f['height']) for f in fs))
  mask_checks=[]
  for f in fs:
   if isinstance(f['masks'],list):
    im=np.asarray(Image.open(ROOT/f['file']))
    mask_checks.append(all(np.all(im[t:b,l:r]==48) for l,t,r,b in f['masks']))
  check(name+':declared_masks_are_48',all(mask_checks),tested=len(mask_checks),exception='original human-check full frame was unmasked; separately repaired and documented' if name=='frame_manifest.json' else None)
 # Reconstruct exact payload bytes, including the one base64 image, without sending a request.
 ledger=readlines(ROOT/'raw/request_ledger.jsonl');raw=readlines(ROOT/'raw/vlm.jsonl');protocol=json.loads((ROOT/'protocols/cap_classifier.json').read_text());errors=[]
 for q in ledger:
  image=ROOT/q['image'];messages=[{'role':m['role'],'content':m['content']} for m in q['messages']];messages[1]['images']=[base64.b64encode(image.read_bytes()).decode()]
  payload={'model':q['model'],'messages':messages,'format':q['schema'],'stream':q['stream'],'think':q['think'],'keep_alive':q['keep_alive'],'options':q['options']}
  ok=(len(messages)==2 and q['messages'][0]['content']==protocol['SYSTEM'] and q['messages'][1]['content']==protocol['PROMPT'] and q['options']==protocol['options'] and q['model_digest']==CFG['model_digest'] and sha(image)==q['image_sha256'] and Image.open(image).size==(224,224) and hashlib.sha256(json.dumps(payload).encode()).hexdigest()==q['request_sha256'])
  if not ok:errors.append(q['item_id'])
 check('full_payload_hashes_prompts_independent_one_image',not errors,items=len(ledger),errors=errors)
 check('ledger_response_one_to_one',[(r['item_id'],r['attempt'],r['request_sha256']) for r in ledger]==[(r['item_id'],r['attempt'],r['request_sha256']) for r in raw])
 check('request_limit',len(ledger)<=400,actual=len(ledger));check('no_retries_or_http_schema_failures',all(r['attempt']==1 and not r['failure'] and r['http_status']==200 for r in raw))
 core=[r for p in (ROOT/'results').glob('regions_*.json') for r in json.loads(p.read_text())];check('330_unique_core_conditions',len(core)==330 and len({r['item_id'] for r in core})==330)
 core_ids={r['item_id'] for r in core}
 check('15_missing_roi_no_requests',sum(r['region_box'] is None for r in core)==15 and sum(r['item_id'] in core_ids for r in ledger)==315)
 check('fair_common_person_boxes',all(len({json.dumps(r['person_box']) for r in core if r['sample_id']==s['id']})==1 for s in ann))
 request_map={r['item_id']:r for r in raw};check('core_numbers_from_raw',all(r['vlm']['number']==request_map[r['item_id']]['number'] for r in core if r['region_box'] is not None))
 corrected=(ROOT/'results/human_mask_summary.json').exists();ctrl=json.loads((ROOT/'results/human_mask_summary.json').read_text()) if corrected else None
 vl=readlines(ROOT/'raw/vision_ledger.jsonl');check('vision_frame_counts',len(vl)==313+2*corrected and len(vl)<=3000,actual=len(vl))
 check('complete_tracking',len(readlines(ROOT/'raw/tracking.jsonl'))==271 and len(readlines(ROOT/'raw/tracking_detector.jsonl'))==271)
 extra=ROOT/'results/resolution_extra.json'
 if extra.exists():
  xr=json.loads(extra.read_text());check('extras_request_counts',len(xr)==24 and len(ledger)==339+(ctrl['new_vlm_requests'] if corrected else 0) and len({r['sample_id'] for r in xr})==12)
  check('jitter96',len(json.loads((ROOT/'results/R2_jitter_extra.json').read_text()))==96)
  check('extra_tracking271',len(readlines(ROOT/'raw/tracking_extra.jsonl'))==271)
 if corrected:
  inp=json.loads((ROOT/'evidence/human_mask_control_input.json').read_text());im=np.asarray(Image.open(ROOT/inp['file']));check('human_mask_correction_input',sha(ROOT/inp['file'])==inp['sha256'] and all(np.all(im[t:b,l:r]==48) for l,t,r,b in inp['masks']))
  reuse=readlines(ROOT/'raw/human_mask_reuse.jsonl');check('human_mask_exact_reuse',all(request_map[r['reused_item_id']]['image_sha256']==r['image_sha256'] for r in reuse),reused=len(reuse))
  cr=json.loads((ROOT/'results/human_mask_control.json').read_text());check('human_mask_complete_separate_control',len(cr)==30 and ctrl['matched_people']==6 and ctrl['new_vlm_requests']+ctrl['reused_requests']+ctrl['unavailable_regions']==30)
  check('human_mask_frozen_before_control',sha(ROOT/'code/human_mask_control.py')==json.loads((ROOT/'evidence/human_mask_control_plan.json').read_text())['code_sha256'])
 for name,n in [('USER_REVIEW.csv',60),('PREVIOUS_GEOMETRY_REVIEW.csv',22),('LEADER_REVIEW.csv',8),('QUESTIONS.csv',12)]:
  with (ROOT/'review'/name).open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
  check('blank_confirmations:'+name,len(rr)==n and all(not r['user_confirmed'] for r in rr),rows=len(rr))
 pose=json.loads((ROOT/'evidence/pose_weights.json').read_text());check('weight_budget_and_hash',pose['bytes']<=1_000_000_000 and sha(ROOT/pose['file'])==pose['sha256'],new_bytes=pose['bytes'])
 missing=[]
 for name in ['MORNING_REPORT_KO.md','README.md','highlights/INDEX_KO.md','review/USER_REVIEW_KO.md']:
  for target in re.findall(r'\]\(([^)]+)\)',(ROOT/name).read_text()):
   if target.startswith(('http:','https:','#')):continue
   target=target.replace('/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117/',str(ROOT)+'/');p=Path(target) if target.startswith('/') else (ROOT/name).parent/target
   if not p.exists():missing.append(str(p))
 check('report_links_exist',not missing,missing=missing)
 gpu=gpu_check();check('disk_above_20GB',gpu['disk_free']>=20_000_000_000,free_bytes=gpu['disk_free'])
 check('research_lock_unheld_and_GPU_compute_idle',not gpu['research_lock_held'] and not gpu['compute_processes'],gpu=gpu)
 out={'time_utc':now(),'all_passed':all(x['passed'] for x in checks),'checks':checks,'scope':'No inference. Original video hashes, frozen files, all reconstructed payloads, input manifests, counts, blank reviews, package link targets. No accuracy ground truth invented.'};dump(ROOT/'evidence/final_verification.json',out)
 print(json.dumps({'checks':len(checks),'all_passed':out['all_passed'],'failed':[c for c in checks if not c['passed']]},ensure_ascii=False));assert out['all_passed']
if __name__=='__main__':main()
