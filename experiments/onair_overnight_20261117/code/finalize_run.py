"""Record final observed totals after verification. Never starts background work."""
import csv,json,subprocess
from common import *
def main():
 v=json.loads((ROOT/'evidence/final_verification.json').read_text());assert v['all_passed']
 raw=readlines(ROOT/'raw/vlm.jsonl');vision=readlines(ROOT/'raw/vision_ledger.jsonl');pids=[]
 for f in sorted((ROOT/'evidence').glob('*pid.json')):
  d=json.loads(f.read_text());pid=d['pid'];stat=Path('/proc')/str(pid)/'stat';state=stat.read_text().rsplit(')',1)[1].split()[0] if stat.exists() else 'absent';pids.append({'record':f.name,'pid':pid,'state':state,'active':state not in ['absent','Z']})
 assert not any(r['active'] for r in pids),'our earlier job still active'
 dump(ROOT/'evidence/processes_end.json',{'time_utc':now(),'jobs':pids,'gpu':gpu_check(),'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)})
 ctrl=json.loads((ROOT/'results/human_mask_control.json').read_text());flat=[]
 for r in ctrl:
  for c in ['hsv','vlm']:
   p=r[c]['number'];flat.append({'sample':r['sample_id'],'method':r['method'],'classifier':c,'reference_number':r['reference_number'],'predicted_number':p,'verdict':'correct' if p==r['reference_number'] else 'abstain' if not p else 'wrong','label_scope':'six human confirmed colours only','input_scope':'post-result full-frame mask correction','state':r[c].get('state'),'reused_item_id':r[c].get('reused_item_id',''),'failure':r[c].get('failure',False)})
 with (ROOT/'results/human_mask_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 summary={'time_utc':now(),'stop_reason':'All preregistered A-F and three extra items processed; D unavailable and recorded. Final input-protocol correction complete. No remaining background job.','stages':{'A':'complete','B':'complete','C':'complete','D':'deferred_missing_saved_time_aligned_official_labels','E':'complete_with_negative_results','F':'complete','extras':'3_complete','human_mask_protocol_repair':'complete_separate_post_result_control'},'vlm_requests':len(raw),'vlm_failures':sum(r['failure'] for r in raw),'vlm_retries':sum(r['attempt']>1 for r in raw),'request_breakdown':{'core':315,'predeclared_resolution':24,'post_result_input_repair':10},'vision_inferences':len(vision),'vision_breakdown':{'C_generic':21,'C_pose':21,'E_generic':271,'input_repair_generic_pose':2},'downloaded_weights_bytes':237034793,'new_video_downloads':0,'paid_api_calls':0,'human_confirmed_colour_observations':6,'new_draft_colour_observations':60,'new_draft_known_colours':56,'human_control_identical_image_response_reuse':18,'primary_hashes_unchanged':True,'final_verification_checks':len(v['checks']),'all_checks_passed':True,'primary_results_preserved':True,'no_future_scheduled_task':True}
 dump(ROOT/'results/RUN_SUMMARY.json',summary)
 status('ALL','complete',reason=summary['stop_reason'],vlm_requests=len(raw),vision_inferences=len(vision),deferred=['D'],background_jobs_remaining=0)
 print('Final run recorded')
if __name__=='__main__':main()
