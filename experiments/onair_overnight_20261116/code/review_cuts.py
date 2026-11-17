import json
import cv2
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,CFG,dump,sha
def main():
 cuts=json.loads((ROOT/'results/cut_candidates.json').read_text());pts=json.loads((ROOT/'evidence/video_pts.json').read_text());wanted={j for c in cuts for j in [c['frame_index']-1,c['frame_index']]};found={};cap=cv2.VideoCapture(CFG['source_video']);i=0
 while wanted:
  ok,bgr=cap.read()
  if not ok:break
  if i in wanted:
   for x1,y1,x2,y2 in CFG['graphics_masks_1280']:bgr[y1:y2,x1:x2]=48
   found[i]=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));wanted.remove(i)
  i+=1
 cap.release();assert not wanted
 dest=ROOT/'review/cuts';dest.mkdir(exist_ok=True);records=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 for k,c in enumerate(cuts):
  board=Image.new('RGB',(1280,400),'#20252b');d=ImageDraw.Draw(board)
  for col,j in enumerate([c['frame_index']-1,c['frame_index']]):
   board.paste(found[j].resize((640,360)),(col*640,0));d.text((col*640+10,366),f'frame{j} PTS{pts[j]:.6f}',font=font,fill='white')
  p=dest/f'cut{k+1}.jpg';board.save(p,quality=95);records.append({**c,'before_pts':pts[c['frame_index']-1],'file':str(p.relative_to(ROOT)),'sha256':sha(p)})
 for page in range(3):
  board=Image.new('RGB',(1280,800),'#20252b')
  for j,r in enumerate(records[page*2:page*2+2]):board.paste(Image.open(ROOT/r['file']),(0,j*400))
  board.save(dest/f'contact{page+1}.jpg',quality=95)
 dump(ROOT/'evidence/cut_review_manifest.json',records)
if __name__=='__main__':main()
