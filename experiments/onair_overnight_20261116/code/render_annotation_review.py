from PIL import Image,ImageDraw,ImageFont,ImageOps
import json
from common import ROOT,dump,sha
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def main():
 ann=json.loads((ROOT/'geometry_annotations.json').read_text());d=ROOT/'review/geometry';d.mkdir(exist_ok=True)
 panels=[];patches=[]
 for f in ann['frames']:
  original=Image.open(ROOT/f['file']).convert('RGB');im=original.copy();draw=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,18)
  caps={c['id']:c for c in f['caps']}
  for c in caps.values():
   x,y=c['center'];draw.ellipse((x-4,y-4,x+4,y+4),outline='yellow',width=2)
  for n in f['noses']:
   x,y=n['point'];c=caps[n['reference_cap_id']];cx,cy=c['center']
   draw.line([(x,y),(cx,cy)],fill='#46ff89',width=2);draw.line((x-5,y-5,x+5,y+5),fill='#ff5b4d',width=3);draw.line((x-5,y+5,x+5,y-5),fill='#ff5b4d',width=3)
   draw.text((cx+8,cy-15),str(n['reference_number']),font=font,fill='white',stroke_width=2,stroke_fill='black')
   crop=(int(min(x,cx)-35),int(min(y,cy)-30),int(max(x,cx)+35),int(max(y,cy)+30))
   patch=original.crop(crop);pd=ImageDraw.Draw(patch);pd.ellipse((x-crop[0]-3,y-crop[1]-3,x-crop[0]+3,y-crop[1]+3),outline='red',width=1)
   patch=ImageOps.contain(patch,(310,195));board=Image.new('RGB',(320,230),'#20252b');board.paste(patch,((320-patch.width)//2,30));bd=ImageDraw.Draw(board);bd.text((5,5),f'{f["pts"]:.0f}s No.{n["reference_number"]} nose {x},{y}',font=font,fill='white');patches.append(board)
  draw.text((250,16),f'{f["pts"]:.1f}s | DRAFT coordinates | direction {f["direction"]}',font=font,fill='white',stroke_width=2,stroke_fill='black')
  im.save(d/(f['id']+'.png'))
  panels.append(im.resize((640,360)))
 for p in range(2):
  board=Image.new('RGB',(1280,720))
  for j,im in enumerate(panels[p*4:p*4+4]):board.paste(im,((j%2)*640,(j//2)*360))
  board.save(d/f'contact_{p+1}.jpg',quality=95)
 for p in range(2):
  board=Image.new('RGB',(960,920))
  for j,im in enumerate(patches[p*12:p*12+12]):board.paste(im,((j%3)*320,(j//3)*230))
  board.save(d/f'patches_{p+1}.jpg',quality=95)
 dump(ROOT/'evidence/annotation_render_final.json',{'annotations_sha256':sha(ROOT/'geometry_annotations.json'),'status':'current frozen annotations rendered; earlier pre-evaluation draft is preserved separately','frames':len(ann['frames']),'noses':sum(len(f['noses']) for f in ann['frames']),'cap_candidates':sum(len(f['caps']) for f in ann['frames'])})
if __name__=='__main__':main()
