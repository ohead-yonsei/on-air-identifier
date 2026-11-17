"""Convert visually drafted patch coordinates to source pixels; no predictions."""
import collections,json,math
from PIL import Image,ImageDraw,ImageFont,ImageOps
from common import *
draft=json.loads((ROOT/'annotation_coordinates_draft.json').read_text());frames={f['id']:f for f in json.loads((ROOT/'frame_manifest.json').read_text())};det={r['frame']:r for r in readlines(ROOT/'raw/detector.jsonl')};out=[];counts=collections.Counter()
for x in draft:
 f=frames[x['frame']];m=next(t for t in json.loads((ROOT/'review/riders'/(x['frame']+'_tiles.json')).read_text()) if t['person_id']==x['person']);ox,oy=m['crop_box'][:2];b=next(q['box'] for q in det[f['id']]['detections'] if q['id']==x['person']);counts[f['id']]+=1
 s={'id':f['id']+'_s'+str(counts[f['id']]),'frame':f['id'],'race':f['race'],'split':f['split'],'view':f['view'],'person_id':x['person'],'person_box':b,'number':x['number'],'label_status':'assistant_draft' if x['number'] else 'unresolved_draft','note':x.get('note',''),'coordinate_status':'assistant rectangle draft, not human-verified segmentation','image_sha256':f['sha256']}
 for key,source in [('cap_box','cap'),('face_box','face'),('torso_box','torso')]:
  v=x.get(source);s[key]=[v[0]+ox,v[1]+oy,v[2]+ox,v[3]+oy] if v else None
 assert s['cap_box'][0]>=0 and s['cap_box'][2]<=f['width'] and s['cap_box'][3]<=f['height'];out.append(s)
old=Path(CFG['old_cap_experiment']);human=json.loads((ROOT/'evidence/human_labels_1010.json').read_text());ids={'G':'p6','H':'p4','I':'p2','J':'p0','K':'p5','L':'p1'}
for s in json.loads((old/'samples.json').read_text()):
 if s['id'] not in human['labels']:continue
 person=ids[s['id']];f=frames['HUMAN_t038_5'];out.append({'id':'HUMAN_'+s['id'],'frame':f['id'],'race':f['race'],'split':'human_check','view':'corner_front','person_id':person,'person_box':next(q['box'] for q in det[f['id']]['detections'] if q['id']==person),'number':human['labels'][s['id']],'label_status':'human_confirmed_colour_only','cap_box':s['cap_box'],'face_box':None,'torso_box':None,'coordinate_status':'original displayed crop; colour/number human confirmed, precise box segmentation not human confirmed','image_sha256':f['sha256'],'source_sample':str(old/'samples.json'),'source_crop_sha256':sha(old/s['cap_file'])})
assert len(out)==66 and len({s['id'] for s in out})==66
ann={'created_utc':now(),'policy':'No classification output seen; person detections predate annotation. All September colour and coordinate labels are assistant drafts. October G-L colours only human confirmed.','samples':out};dump(ROOT/'annotations.json',ann)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);colours=['unknown','white','yellow','red','black','blue','green','brown','pink','purple','sky','blue-white stripes']
tiles=[]
for s in out:
 f=frames[s['frame']];im=Image.open(ROOT/f['file']).convert('RGB');patch=im.crop(s['cap_box']);tile=Image.new('RGB',(240,245),'#182b35');part=ImageOps.contain(patch,(210,175));tile.paste(part,((240-part.width)//2,60));d=ImageDraw.Draw(tile);d.text((5,3),s['id'],font=font,fill='white');d.text((5,24),f'{s["number"]}: {colours[s["number"]]} | {patch.width}x{patch.height}',font=small,fill='#ffdf75');d.text((5,42),s['label_status'],font=small,fill='#c3d9df');tiles.append(tile)
for k in range(0,len(tiles),24):
 part=tiles[k:k+24];sheet=Image.new('RGB',(1440,245*math.ceil(len(part)/6)),(35,35,35))
 for j,t in enumerate(part):sheet.paste(t,((j%6)*240,(j//6)*245))
 sheet.save(ROOT/'review'/f'draft_caps_{k//24+1}.jpg',quality=97)
for race in sorted({s['race'] for s in out}):
 ss=[s for s in out if s['race']==race];fis=list(dict.fromkeys(s['frame'] for s in ss));panels=[]
 for fid in fis:
  f=frames[fid];im=Image.open(ROOT/f['file']).convert('RGB');d=ImageDraw.Draw(im)
  for s in ss:
   if s['frame']!=fid:continue
   d.rectangle(s['cap_box'],outline='#40ff80',width=3);d.text((s['cap_box'][0],max(0,s['cap_box'][1]-24)),f'{s["id"].rsplit("_",1)[-1]}: {s["number"]} {colours[s["number"]]}',font=font,fill='yellow',stroke_fill='black',stroke_width=2)
  panels.append(im.resize((960,round(im.height*960/im.width))))
 sheet=Image.new('RGB',(960,sum(i.height for i in panels)+45),'#182b35');ImageDraw.Draw(sheet).text((10,10),f'{race}: DRAFT cap boxes/colours; please review',font=font,fill='white');y=45
 for im in panels:sheet.paste(im,(0,y));y+=im.height
 sheet.save(ROOT/'review'/f'{race}_draft.jpg',quality=95)
dump(ROOT/'evidence/annotation_preview_stats.json',{'samples':len(out),'colour_counts':dict(collections.Counter(s['number'] for s in out)),'races':len({s['race'] for s in out}),'new_races':10,'new_frames':20,'human_confirmed_samples':6,'unresolved_samples':sum(s['number']==0 for s in out),'face_rectangles':sum(s['face_box'] is not None for s in out),'torso_rectangles':sum(s['torso_box'] is not None for s in out)})
print(json.dumps(json.loads((ROOT/'evidence/annotation_preview_stats.json').read_text())))
