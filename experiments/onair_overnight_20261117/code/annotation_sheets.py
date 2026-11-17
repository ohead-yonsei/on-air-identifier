"""Make unmarked pixels readable for manual draft annotation; no inference."""
from PIL import Image,ImageDraw,ImageFont
import math,json
from common import *
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',11)
for r in readlines(ROOT/'raw/detector.jsonl'):
 im=Image.open(ROOT/r['file']).convert('RGB');meta=json.loads((ROOT/'review/riders'/(r['frame']+'_tiles.json')).read_text());tiles=[]
 for m in meta:
  box=m['crop_box'];patch=im.crop(box);scale=min(260/patch.width,220/patch.height);nw,nh=round(patch.width*scale),round(patch.height*scale);tile=Image.new('RGB',(320,290),'#182a35');tile.paste(patch.resize((nw,nh)),(40,50));d=ImageDraw.Draw(tile);d.text((5,2),f'{m["person_id"]} origin {box[:2]}',font=font,fill='white');d.text((5,23),f'native {patch.size} scale {scale:.3f}',font=small,fill='white')
  step=10 if patch.width<100 else 20
  for x in range(0,patch.width+1,step):
   xx=40+x*scale;d.line((xx,45,xx,50),fill='yellow');d.text((xx-5,33),str(x),font=small,fill='yellow')
  for y in range(0,patch.height+1,step):
   yy=50+y*scale;d.line((35,yy,40,yy),fill='yellow');d.text((10,yy-5),str(y),font=small,fill='yellow')
  tiles.append(tile)
 sheet=Image.new('RGB',(1280,290*math.ceil(len(tiles)/4)),(30,30,30))
 for k,tile in enumerate(tiles):sheet.paste(tile,((k%4)*320,(k//4)*290))
 sheet.save(ROOT/'review/riders'/(r['frame']+'_clean.jpg'),quality=97)
print('saved clean annotation sheets',len(readlines(ROOT/'raw/detector.jsonl')))
