import bisect,json
from pathlib import Path
import cv2
from common import *
pts=json.loads((Path(CFG['old_experiment'])/'evidence/video_pts.json').read_text());targets={bisect.bisect_right(pts,round(31+k*.1,1)+1e-8)-1:round(31+k*.1,1) for k in range(271)};cap=cv2.VideoCapture(CFG['old_video']);start=min(targets);cap.set(cv2.CAP_PROP_POS_FRAMES,start);dest=ROOT/'tracking/frames';dest.mkdir(parents=True,exist_ok=True);out=[];cv2.setNumThreads(4)
for idx in range(start,max(targets)+1):
 ok,im=cap.read();assert ok
 if idx not in targets:continue
 assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(idx+1))<.01
 assert im.shape[:2]==(720,1280)
 for x1,y1,x2,y2 in [[0,0,240,110],[0,610,1280,720]]:im[y1:y2,x1:x2]=48
 t=targets[idx];fid=f'track_{round(t*10):04}';path=dest/(fid+'.png');cv2.imwrite(str(path),im);out.append({'id':fid,'file':str(path.relative_to(ROOT)),'target_pts':t,'actual_pts':pts[idx],'decoded_index':idx,'sha256':sha(path),'width':1280,'height':720,'source_video':CFG['old_video'],'masks':[[0,0,240,110],[0,610,1280,720]]})
cap.release();assert len(out)==271 and all(x['actual_pts']<=x['target_pts']+1e-8 for x in out);dump(ROOT/'tracking_frame_manifest.json',out);status('E_extract','complete',frames=271,first_pts=out[0]['actual_pts'],last_pts=out[-1]['actual_pts'])
