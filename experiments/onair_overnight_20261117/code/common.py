from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'config.json').read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def append(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def readlines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()] if Path(p).exists() else []
def status(stage,state,**info):
 d={'time_utc':now(),'time_kst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'stage':stage,'state':state,**info};dump(ROOT/'status.json',d);append(ROOT/'journal.jsonl',d);print(json.dumps(d,ensure_ascii=False),flush=True)
def guard():
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['deadline_utc'].replace('Z','+00:00')),'deadline reached'
 assert shutil.disk_usage(ROOT).free>=20_000_000_000,'disk below20GB'
def gpu_check():
 guard();lock=Path('/home/ohead/icip2027_research_gpu.lock');held=[]
 if lock.exists():
  inode=str(lock.stat().st_ino)
  held=[s for s in Path('/proc/locks').read_text().splitlines() if any(t.endswith(':'+inode) for t in s.split())]
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
 # Never signal or unload another process. Call before loading our model.
 return {'time_utc':now(),'research_lock_exists':lock.exists(),'research_lock_held':bool(held),'compute_processes':raw,'disk_free':shutil.disk_usage(ROOT).free}
def wait_gpu():
 while True:
  g=gpu_check();append(ROOT/'evidence/gpu_checks.jsonl',g)
  if not g['research_lock_held'] and not g['compute_processes']:return
  status('gpu','waiting',reason=g);time.sleep(30)
def count_inference(kind,frame):
 ledger=ROOT/'raw/vision_ledger.jsonl';rows=readlines(ledger);assert len(rows)<CFG['max_vision_frames'];append(ledger,{'time_utc':now(),'kind':kind,'frame':frame})
