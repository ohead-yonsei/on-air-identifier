import bisect,json,shutil
import cv2
from common import *
cv2.setNumThreads(4)
sel=json.loads((ROOT/'selection.json').read_text());vs={x['race']:x for x in json.loads((ROOT/'evidence/video_sources.json').read_text())['videos']};out=[]
for s in sel['frames']:
 race=s['race'];video=ROOT/vs[race]['file'];pts=json.loads((ROOT/'evidence'/f'{race}_pts.json').read_text());i=bisect.bisect_right(pts,s['target_pts']+1e-8)-1
 cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();assert ok and abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(i+1))<.01;cap.release()
 for x1,y1,x2,y2 in sel['race_masks'][race]:im[y1:y2,x1:x2]=48
 path=ROOT/'frames'/(s['id']+'.png');cv2.imwrite(str(path),im)
 out.append({**s,'source_video':str(video),'source_sha256':vs[race]['sha256'],'file':str(path.relative_to(ROOT)),'sha256':sha(path),'decoded_index':i,'actual_pts':pts[i],'width':im.shape[1],'height':im.shape[0],'masks':sel['race_masks'][race]})
# Exact previously reviewed image, copied without altering pixels.
old=Path(CFG['old_cap_experiment']);sample=[s for s in json.loads((old/'samples.json').read_text()) if s['id']=='G'][0]
src=old/sample['source_file'];assert sha(src)==sample['source_sha256'];target=ROOT/'frames/HUMAN_t038_5.png';shutil.copy2(src,target)
out.append({'id':'HUMAN_t038_5','race':'20261010_R10','split':'human_check','view':'corner_front','actual_pts':38.5,'target_pts':38.5,'source_video':CFG['old_video'],'source_image':str(src),'file':str(target.relative_to(ROOT)),'sha256':sha(target),'width':1280,'height':720,'masks':'unchanged exact previously reviewed source image'})
dump(ROOT/'frame_manifest.json',out);status('B_extract','complete',frames=len(out),future_frame_count=sum(x['actual_pts']>x['target_pts']+1e-8 for x in out))
