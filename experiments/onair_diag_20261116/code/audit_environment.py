import json, pathlib, subprocess, urllib.request, datetime, os
ROOT=pathlib.Path(__file__).resolve().parents[1]
E=ROOT/'evidence'; E.mkdir(exist_ok=True)
def cmd(args, timeout=60):
 try:
  r=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
  return {'args':args,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 except Exception as e: return {'args':args,'error':str(e)}
def api(path, body=None):
 req=urllib.request.Request('http://127.0.0.1:11435/api/'+path,data=json.dumps(body).encode() if body else None,headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(req,timeout=30))
r={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(ROOT),'checks':[]}
for args in [['hostname'],['uname','-a'],['cat','/etc/os-release'],['nvidia-smi'],['ffmpeg','-version'],['ffprobe','-version'],['/home/ohead/ai-workspace/.venv/bin/python','-c','import sys,torch,importlib.metadata as m; print(sys.executable,sys.version); print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print({p:m.version(p) for p in ["transformers","accelerate","Pillow"]})']]:
 r['checks'].append(cmd(args))
for name in ['version','tags','ps']:
 try: r['ollama_'+name]=api(name)
 except Exception as e: r['ollama_'+name]={'error':str(e)}
try: r['selected_model_show']=api('show',{'model':'qwen3.5:9b'})
except Exception as e: r['selected_model_show']={'error':str(e)}
(E/'environment.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps({k:r[k] for k in ['time','root','ollama_version']},ensure_ascii=False))
# rg is not installed on this server; bounded find is the fallback.
find=['find','/home/ohead','/mnt','/media','/data','-maxdepth','8','-type','d','(','-name','.cache','-o','-name','.local','-o','-name','.config','-o','-name','.ssh','-o','-name','.venv','-o','-name','.ollama','-o','-name','node_modules','-o','-name','.git','-o','-name','snap',')','-prune','-o','-type','f','(','-iname','*onair*','-o','-iname','*on-air*','-o','-name','scenes.json','-o','-name','owners.json','-o','-name','visible.json','-o','-name','run_ab2.py','-o','-iname','*.mp4','-o','-iname','*.mkv',')','-print']
r=cmd(find,timeout=45)
(E/'server_discovery.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps(r,ensure_ascii=False))
