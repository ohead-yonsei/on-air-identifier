from pathlib import Path
import bisect,csv,hashlib,json,subprocess,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'config.json').read_text())
def write(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def stamp():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def main():
 for d in ['frames','review','evidence','results','raw','highlights']:(ROOT/d).mkdir(exist_ok=True)
 video=Path(CFG['source_video']);sha=hashlib.file_digest(video.open('rb'),'sha256').hexdigest();assert sha==CFG['source_sha256']
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)]))
 pts=[float(x['best_effort_timestamp_time']) for x in probe['frames']];assert pts==sorted(pts)
 write('evidence/video_pts.json',pts)
 targets=[float(t) for t in range(0,int(pts[-1])+1)]
 selected={bisect.bisect_right(pts,t+1e-8)-1:t for t in targets if t>=pts[0]}
 cap=cv2.VideoCapture(str(video));last=None;last_hist=None;rows=[];saved=[];i=0
 start=time.perf_counter()
 while True:
  ok,bgr=cap.read()
  if not ok:break
  assert i<len(pts)
  for x1,y1,x2,y2 in CFG['graphics_masks_1280']:bgr[y1:y2,x1:x2]=48
  small=cv2.resize(bgr,(320,180),interpolation=cv2.INTER_AREA)
  hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV);hist=cv2.calcHist([hsv],[0,1],None,[24,16],[0,180,0,256]);cv2.normalize(hist,hist)
  if last is not None:
   mad=float(np.mean(np.abs(small.astype(np.float32)-last.astype(np.float32)))/255)
   bhat=float(cv2.compareHist(hist,last_hist,cv2.HISTCMP_BHATTACHARYYA))
   rows.append({'frame_index':i,'pts':pts[i],'mad':mad,'hist_bhattacharyya':bhat,'candidate':bhat>0.35 or mad>0.18})
  if i in selected:
   p=ROOT/'frames'/f't{selected[i]:06.2f}.jpg';cv2.imwrite(str(p),bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
   saved.append({'target_seconds':selected[i],'actual_pts':pts[i],'decoded_index':i,'file':str(p.relative_to(ROOT)),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'width':bgr.shape[1],'height':bgr.shape[0]})
  last=small;last_hist=hist;i+=1
 cap.release();assert i==len(pts)
 with (ROOT/'results/cut_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 candidates=[]
 for r in rows:
  if not r['candidate']:continue
  if candidates and r['pts']-candidates[-1]['pts']<0.35:
   if r['hist_bhattacharyya']+r['mad']>candidates[-1]['hist_bhattacharyya']+candidates[-1]['mad']:candidates[-1]=r
  else:candidates.append(r)
 write('results/cut_candidates.json',candidates);write('frame_manifest.json',saved)
 show=[s for s in saved if int(s['target_seconds'])%3==0]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 for page in range((len(show)+15)//16):
  chunk=show[page*16:(page+1)*16];im=Image.new('RGB',(1280,840),'#20252b');d=ImageDraw.Draw(im)
  for j,s in enumerate(chunk):
   x=j%4*320;y=j//4*210;im.paste(Image.open(ROOT/s['file']).resize((320,180)),(x,y));d.text((x+6,y+182),f'PTS {s["actual_pts"]:.3f}s',font=font,fill='white')
  im.save(ROOT/'review'/f'survey_{page+1}.jpg',quality=95)
 write('evidence/survey_execution.json',{'finished_at':stamp(),'decoded_frames':i,'saved_frames':len(saved),'cut_candidates':len(candidates),'elapsed_s':time.perf_counter()-start,'source_sha256':sha,'python_tool':'OpenCV sequential decode; timestamps from ffprobe frame list','audio_used':False,'mask':CFG['graphics_masks_1280']})
 print(json.dumps({'saved':len(saved),'candidates':candidates,'elapsed_s':time.perf_counter()-start},indent=2),flush=True)
if __name__=='__main__':main()
