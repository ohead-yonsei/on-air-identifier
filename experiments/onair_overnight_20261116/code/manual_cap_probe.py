import base64,hashlib,json,time,urllib.request
from types import SimpleNamespace
from PIL import Image,ImageOps,ImageDraw,ImageFont
from common import ROOT,CFG,sha,dump,append,status,check_deadline,now
def main():
 cfg=json.loads((ROOT/'manual_cap_probe.json').read_text());protocol=json.loads((ROOT/'protocols/cap_classifier.json').read_text());m=SimpleNamespace(**protocol)
 opts=protocol['options'];frames={s['id']:s for s in json.loads((ROOT/'selection.json').read_text())['frames']};dest=ROOT/'manual_cap';dest.mkdir(exist_ok=True);out=ROOT/'raw/manual_cap.jsonl';done={r['item_id'] for r in map(json.loads,out.read_text().splitlines())} if out.exists() else set();panels=[]
 for fid,box in cfg['boxes_1280'].items():
  ip=dest/(fid+'.png');im=Image.open(ROOT/frames[fid]['file']).convert('RGB').crop(box);patch=ImageOps.contain(im,(224,224),Image.Resampling.LANCZOS);can=Image.new('RGB',(224,224),(96,96,96));can.paste(patch,((224-patch.width)//2,(224-patch.height)//2));can.save(ip);panel=Image.new('RGB',(240,260),'#20252b');panel.paste(can,(8,30));ImageDraw.Draw(panel).text((10,5),fid+' manual tight cap',font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16),fill='white');panels.append(panel)
  item=fid+'_manual_tight_cap'
  if item in done:continue
  check_deadline();ledger=ROOT/'raw/request_ledger.jsonl';assert len(ledger.read_text().splitlines())<CFG['max_vlm_requests_including_failures']
  messages=[{'role':'system','content':m.SYSTEM},{'role':'user','content':m.PROMPT,'images':[base64.b64encode(ip.read_bytes()).decode()]}];body={'model':CFG['model'],'messages':messages,'format':m.SCHEMA,'stream':False,'think':False,'keep_alive':'10m','options':opts};data=json.dumps(body).encode()
  rec={'item_id':item,'frame':fid,'stage':'manual_tight_cap','model':CFG['model'],'model_digest':CFG['model_digest'],'options':opts,'messages':[{'role':x['role'],'content':x['content']} for x in messages],'schema':m.SCHEMA,'image':str(ip.relative_to(ROOT)),'image_sha256':sha(ip),'source_box':box,'request_sha256':hashlib.sha256(data).hexdigest(),'started_at':now(),'attempt':1,'failure':False}
  append(ledger,rec);s=time.perf_counter();raw=None
  try:
   req=urllib.request.Request(CFG['endpoint']+'/api/chat',data=data,headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=180) as rr:raw=rr.read().decode();rec['http_status']=rr.status
   parsed=json.loads(json.loads(raw)['message']['content']);assert parsed['color_class'] in m.CLASSES;rec.update(raw_http_response=raw,parsed=parsed,number=m.CLASSES.index(parsed['color_class']))
  except Exception as e:rec.update(failure=True,error=repr(e),raw_http_response=raw)
  rec['http_elapsed_ms']=(time.perf_counter()-s)*1000;append(out,rec);status('manual_cap_probe','running',frame=fid,failure=rec['failure'])
 board=Image.new('RGB',(960,520),'#20252b')
 for i,im in enumerate(panels):board.paste(im,((i%4)*240,(i//4)*260))
 board.save(ROOT/'review/manual_cap_probe.jpg',quality=95);status('manual_cap_probe','complete',cases=len(panels))
if __name__=='__main__':main()
