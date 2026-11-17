"""Paired region diagnostics; human accuracy and draft agreement kept separate."""
import collections,csv,json,statistics
from common import *
def avg(xs):return sum(xs)/len(xs) if xs else None
def med(xs):return statistics.median(xs) if xs else None
def main():
 ann={s['id']:s for s in json.loads((ROOT/'annotations.json').read_text())['samples']};rows=[]
 for p in sorted((ROOT/'results').glob('regions_*.json')):
  if 'extra' not in p.name:rows+=json.loads(p.read_text())
 assert len({r['item_id'] for r in rows})==len(rows)
 flat=[]
 for r in rows:
  for clf in ['hsv','vlm']:
   pred=r.get(clf);label=r['reference_number'];eligible=label>0
   if pred is None:continue
   n=pred.get('number',0);verdict='unverified_reference' if not eligible else 'abstain' if n==0 else 'match' if n==label else 'mismatch'
   flat.append({'sample_id':r['sample_id'],'race':r['race'],'frame':r['frame'],'split':r['split'],'method':r['method'],'classifier':clf,'view':r['view'],'size_group':r['size_group'],'reference_status':r['reference_status'],'reference_number':label,'predicted_number':n,'verdict':verdict,'state':pred.get('state'),'high_evidence_mismatch':bool(clf=='hsv' and pred.get('state')=='high_evidence' and verdict=='mismatch'),'asserted_mismatch':verdict=='mismatch','failure':pred.get('failure',False),'cap_iou':r['cap_iou'],'cap_center_included':r['cap_center_included'],'region_available':r['region_box'] is not None,'cap_reference_reliable':ann[r['sample_id']]['cap_box_reliability']!='uncertain','outside_cap_box_fraction':r['outside_cap_box_fraction'],'face_box_overlap_fraction':r['face_box_overlap_fraction'],'torso_box_overlap_fraction':r['torso_box_overlap_fraction'],'region_ms':r['region_ms'],'classification_ms':pred.get('http_elapsed_ms',pred.get('elapsed_ms',0)),'region_plus_classification_ms':r['region_ms']+pred.get('http_elapsed_ms',pred.get('elapsed_ms',0))})
 def aggregate(g):
  labelled=[r for r in g if r['reference_number']>0];good=[r for r in labelled if r['verdict']=='match'];bad=[r for r in labelled if r['verdict']=='mismatch'];absn=[r for r in labelled if r['verdict']=='abstain'];roi=[r for r in g if r['cap_reference_reliable']];available=[r for r in roi if r['region_available']]
  return {'observations':len(g),'races':len({r['race'] for r in g}),'frames':len({r['frame'] for r in g}),'race_number_entities_draft':len({(r['race'],r['reference_number']) for r in labelled}),'labelled_observations':len(labelled),'unresolved_reference':len(g)-len(labelled),'match':len(good),'mismatch':len(bad),'abstain':len(absn),'match_rate_all_labelled':len(good)/len(labelled) if labelled else None,'match_rate_when_asserted':len(good)/(len(good)+len(bad)) if good or bad else None,'high_evidence_mismatch':sum(r['high_evidence_mismatch'] for r in g) if g and g[0]['classifier']=='hsv' else None,'asserted_mismatch':len(bad),'region_available':sum(r['region_available'] for r in g),'roi_reference_count':len(roi),'cap_iou_mean_all_including_no_region':avg([r['cap_iou'] for r in roi]),'cap_iou_mean_available':avg([r['cap_iou'] for r in available]),'cap_center_included':sum(r['cap_center_included'] for r in roi),'outside_cap_box_fraction_mean':avg([r['outside_cap_box_fraction'] for r in available if r['outside_cap_box_fraction'] is not None]),'face_proxy_n':sum(r['face_box_overlap_fraction'] is not None for r in available),'face_proxy_mean':avg([r['face_box_overlap_fraction'] for r in available if r['face_box_overlap_fraction'] is not None]),'torso_proxy_n':sum(r['torso_box_overlap_fraction'] is not None for r in available),'torso_proxy_mean':avg([r['torso_box_overlap_fraction'] for r in available if r['torso_box_overlap_fraction'] is not None]),'median_region_ms':med([r['region_ms'] for r in g]),'median_classification_ms_available':med([r['classification_ms'] for r in g if r['region_available']]),'median_region_plus_classification_ms_available':med([r['region_plus_classification_ms'] for r in g if r['region_available']]),'failures':sum(r['failure'] for r in g)}
 groups={}
 for keyfields in [('split','method','classifier'),('split','method','classifier','view'),('split','method','classifier','size_group'),('split','method','classifier','race')]:
  grouped=collections.defaultdict(list)
  for r in flat:grouped[tuple(r[k] for k in keyfields)].append(r)
  groups['by_'+'_'.join(keyfields)]=[{**dict(zip(keyfields,key)),**aggregate(g)} for key,g in grouped.items()]
 confusion=collections.Counter((r['split'],r['method'],r['classifier'],r['reference_number'],r['predicted_number']) for r in flat if r['reference_number']>0)
 paired=[]
 for split in sorted({r['split'] for r in flat}):
  for clf in ['hsv','vlm']:
   byid=collections.defaultdict(dict)
   for r in flat:
    if r['split']==split and r['classifier']==clf and r['reference_number']>0:byid[r['sample_id']][r['method']]=r
   for method in CFG['methods']:
    vals=[(x['R0_35'],x[method]) for x in byid.values() if 'R0_35' in x and method in x];paired.append({'split':split,'classifier':clf,'method':method,'baseline':'R0_35','n':len(vals),'became_match':sum(a['verdict']!='match' and b['verdict']=='match' for a,b in vals),'lost_match':sum(a['verdict']=='match' and b['verdict']!='match' for a,b in vals),'both_match':sum(a['verdict']=='match' and b['verdict']=='match' for a,b in vals)})
 raw=readlines(ROOT/'raw/vlm.jsonl');vision=readlines(ROOT/'raw/vision_ledger.jsonl');summary={'created_utc':now(),'semantic_policy':'human_check match means colour accuracy on six human reviewed crops; other splits mean assistant draft agreement, not accuracy. Unknown references excluded from agreement denominators.','groups':groups,'paired_changes':paired,'model_requests':len(raw),'model_failures':sum(r['failure'] for r in raw),'retries':sum(r['attempt']>1 for r in raw),'vision_inferences':len(vision),'classification_http_seconds':sum(r['http_elapsed_ms'] for r in raw)/1000,'draft_label_hash':sha(ROOT/'annotations.json'),'legacy_v4_calibration_warning':'R1/R2/R4 were involved in the historical v4 colour calibration; no historically unseen-race claim for v4.'};dump(ROOT/'results/C_SUMMARY.json',summary)
 for name,data in [('condition_comparison.csv',flat),('condition_summary.csv',groups['by_split_method_classifier']),('paired_changes.csv',paired),('colour_confusion.csv',[{'split':k[0],'method':k[1],'classifier':k[2],'reference':k[3],'prediction':k[4],'n':n} for k,n in confusion.items()])]:
  if data:
   with (ROOT/'results'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 print(json.dumps({'rows':len(flat),'model_requests':len(raw),'summary':groups['by_split_method_classifier']},ensure_ascii=False))
if __name__=='__main__':main()
