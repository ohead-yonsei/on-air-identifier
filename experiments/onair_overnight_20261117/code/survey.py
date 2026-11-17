import bisect,json,math,subprocess
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import *
cv2.setNumThreads(4)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
sources=json.loads((ROOT/'evidence/video_sources.json').read_text())['videos']
out=[]
for v in sources:
 guard();race=v['race'];src=ROOT/v['file'];pp=ROOT/'evidence'/f'{race}_pts.json'
 if pp.exists():pts=json.loads(pp.read_text())
 else:
  raw=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(src)])
  pts=[float(x['best_effort_timestamp_time']) for x in json.loads(raw)['frames']];dump(pp,pts)
 targets=[t for t in range(10,161,10) if t<pts[-1]];cap=cv2.VideoCapture(str(src));tiles=[]
 for t in targets:
  i=bisect.bisect_right(pts,t+1e-8)-1;cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=cap.read();assert ok and abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(i+1))<.01
  path=ROOT/'survey'/f'{race}_t{t:03}.jpg';cv2.imwrite(str(path),bgr,[cv2.IMWRITE_JPEG_QUALITY,95]);im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((640,360));tile=Image.new('RGB',(640,392),'#1a2630');tile.paste(im,(0,32));ImageDraw.Draw(tile).text((10,4),f'{race} PTS {pts[i]:.6f}',fill='white',font=font);tiles.append(tile)
  out.append({'race':race,'target':t,'actual_pts':pts[i],'decoded_index':i,'file':str(path.relative_to(ROOT)),'sha256':sha(path),'size':[bgr.shape[1],bgr.shape[0]],'survey_only':True})
 cap.release();sheet=Image.new('RGB',(1280,392*math.ceil(len(tiles)/2)),(40,40,40))
 for k,tile in enumerate(tiles):sheet.paste(tile,((k%2)*640,(k//2)*392))
 sheet.save(ROOT/'survey'/f'{race}_contact.jpg',quality=92)
 status('B_survey','running',race=race,frames=len(tiles))
dump(ROOT/'evidence/survey_manifest.json',out);status('B_survey','complete',races=10,frames=len(out))
