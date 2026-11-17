"""Bounded detached batch; children run sequentially, each stage logs independently."""
import os,subprocess,sys
from common import ROOT,status,check_deadline,now,dump
def main():
 dump(ROOT/'batch_process.json',{'pid':os.getpid(),'started_at':now(),'python':sys.executable,'root':str(ROOT),'detached':True})
 jobs=[['run_detector.py'],['run_vlm.py','--task','view'],['run_vlm.py','--task','view','--mirror'],['run_vlm.py','--task','localize'],['run_vlm.py','--task','conditioned'],['run_geometry.py']]
 results=[]
 for args in jobs:
  check_deadline();stage=' '.join(args);status('batch','stage_start',command=stage)
  r=subprocess.run([sys.executable,str(ROOT/'code'/args[0]),*args[1:]],cwd=ROOT)
  results.append({'command':stage,'exit_code':r.returncode});dump(ROOT/'batch_results.json',results)
  status('batch','stage_end',command=stage,exit_code=r.returncode)
  if r.returncode!=0:status('batch','stopped_on_failure',results=results);return
 status('batch','complete',results=results)
if __name__=='__main__':main()
