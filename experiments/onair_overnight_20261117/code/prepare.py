import json,os,platform,shutil,subprocess,urllib.request
from pathlib import Path
from common import *
guard()
import torch,torchvision,cv2,numpy,scipy,PIL
out={'time_utc':now(),'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_build':torch.version.cuda,'cv2':cv2.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'pillow':PIL.__version__,'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'disk':dict(zip(['total','used','free'],shutil.disk_usage(ROOT))),'gpu_check':gpu_check()}
for tag,url in [('ollama_tags','/api/tags'),('ollama_version','/api/version')]:
 with urllib.request.urlopen(CFG['endpoint']+url) as r:out[tag]=json.loads(r.read())
assert any(m['name']==CFG['model'] and m['digest']==CFG['model_digest'] for m in out['ollama_tags']['models'])
dump(ROOT/'evidence/environment_start.json',out)
before=json.loads((ROOT/'evidence/video_sources_original.json').read_text());after=json.loads((ROOT/'evidence/video_sources.json').read_text())
b={x['race']:x for x in before['videos']};checks=[]
for v in after['videos']:
 p=v['ffprobe'];ok=v['status']=='ok' and v['sha256']==b[v['race']]['sha256'] and p['width']==1920 and p['height']==1080 and p['avg_fps']==60 and p['audio_streams']==0
 checks.append({'race':v['race'],'passed':ok,'sha256':v['sha256']})
assert len(checks)==10 and all(x['passed'] for x in checks)
dump(ROOT/'evidence/video_reverification.json',{'time_utc':now(),'checks':checks,'originals_unchanged':True})
cache=ROOT/'model_cache/checkpoints';cache.mkdir(parents=True,exist_ok=True)
old=Path('/home/ohead/ai-workspace/projects/onair_overnight_20261116/model_cache/checkpoints')
for p in old.glob('fasterrcnn_resnet50_fpn_v2*'):
 target=cache/p.name
 if not target.exists():shutil.copy2(p,target)
 dump(ROOT/'evidence/copied_detector_weight.json',{'source':str(p),'target':str(target.relative_to(ROOT)),'sha256':sha(target),'bytes':target.stat().st_size,'new_download_bytes':0})
found=[]
for base in [Path('/home/ohead/inputs_0906'),Path('/home/ohead/ai-workspace/projects/inputs_0906'),ROOT/'inputs_0906']:
 if base.exists():found += [str(x) for x in base.rglob('*') if x.is_file()]
dump(ROOT/'evidence/expert_source_search.json',{'searched_paths':['Mac inputs_0906 (absent)']+['/home/ohead/inputs_0906','/home/ohead/ai-workspace/projects/inputs_0906',str(ROOT/'inputs_0906')],'files':found,'project_filename_search':'find performed for scenes.json owners.json visible.json; no API or credentials searched'})
status('A','complete',videos_verified=10,gpu=out['gpu_check'],expert_files=len(found),plan_sha256=sha(ROOT/'PLAN_KO.md'))
