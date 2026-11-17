from pathlib import Path
import datetime,hashlib,json,os,time
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'config.json').read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def append(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def status(stage,state,**info):
 obj={'time_utc':now(),'time_kst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'stage':stage,'state':state,**info}
 dump(ROOT/'status.json',obj);append(ROOT/'journal.jsonl',obj);print(json.dumps(obj,ensure_ascii=False),flush=True)
def check_deadline():
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(CFG['deadline_utc'].replace('Z','+00:00')):raise RuntimeError('Configured morning deadline reached')
