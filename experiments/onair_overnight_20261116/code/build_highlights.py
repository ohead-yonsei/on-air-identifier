"""Analytical crops/overlays only. No generated or inpainted evidence pixels."""
import csv,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
from common import ROOT,dump
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):return ImageFont.truetype(FONT,n)
def text(d,p,s,fill='white',size=18):d.text(p,s,font=font(size),fill=fill,stroke_width=2,stroke_fill='#14202a')
def arrow(d,p,v,length=70,color='#50e4ff'):
 norm=math.hypot(*v);u=[x/norm for x in v];q=[p[i]+u[i]*length for i in [0,1]];perp=[-u[1],u[0]]
 d.line([tuple(p),tuple(q)],fill=color,width=4);d.polygon([tuple(q),(q[0]-u[0]*14+perp[0]*7,q[1]-u[1]*14+perp[1]*7),(q[0]-u[0]*14-perp[0]*7,q[1]-u[1]*14-perp[1]*7)],fill=color)
def contain(im,size):return ImageOps.contain(im,size,Image.Resampling.LANCZOS)
def main():
 dest=ROOT/'highlights';dest.mkdir(exist_ok=True);ann={f['id']:f for f in json.loads((ROOT/'geometry_annotations.json').read_text())['frames']};vlm={r['item_id']:r for r in map(json.loads,(ROOT/'raw/vlm.jsonl').read_text().splitlines())};mod={r['item_id']:r for r in json.loads((ROOT/'results/modular_pipeline.json').read_text())};manual={r['frame']:r for r in map(json.loads,(ROOT/'raw/manual_cap.jsonl').read_text().splitlines())};flow={r['frame']:r for r in json.loads((ROOT/'results/flow_01_affine_direction.json').read_text())}
 manifest=[]
 # 1: One uninterrupted shot can contain different view directions.
 board=Image.new('RGB',(1440,410),'#14202a');d=ImageDraw.Draw(board)
 text(d,(20,12),'ONE CONTINUOUS SHOT: 32.733s - 43.067s',size=25)
 for j,fid in enumerate(['t034','t037','t042']):
  f=ann[fid];im=Image.open(ROOT/f['file']).resize((480,270));board.paste(im,(j*480,60));arrow(d,(j*480+260,140),f['direction'],color='#49fa8f');p=vlm[fid+'_view']['parsed'];text(d,(j*480+12,337),f'{f["pts"]:.0f}s | manual view: {f["view"]}',size=18);text(d,(j*480+12,367),f'9B: {p["view"]}, {p["direction"]}',fill='#ffbe7b',size=18)
 board.save(dest/'01_same_shot_view_changes.jpg',quality=95);manifest.append({'file':'highlights/01_same_shot_view_changes.jpg','claim':'Within one shot, view changes from profile through frontal to opposite profile. Arrows are manual heuristic axes, not calibrated motion.'})
 # 2: Same model and prompt; automatic broad ROI vs manual tight helmet crop.
 board=Image.new('RGB',(1150,960),'#14202a');d=ImageDraw.Draw(board);text(d,(20,10),'SAME FRAMES / SAME CAP CLASSIFIER / DIFFERENT REGION',size=24)
 for j,fid in enumerate(['t031','t034','t037','t048']):
  y=70+j*215;auto=mod[fid+'_manual_axis'];tight=manual[fid];im=Image.open(ROOT/auto['image']);ti=Image.open(ROOT/tight['image']);board.paste(im.resize((190,190)),(380,y));board.paste(ti.resize((190,190)),(780,y));context=Image.open(ROOT/auto['image'].replace('.png','_overlay.jpg')).resize((320,180));board.paste(context,(20,y));text(d,(20,y+183),fid+' | selected horse/rider',size=16);text(d,(590,y+55),'->',size=30);text(d,(380,y+185),'auto: '+str(auto['parsed']['color_class']),fill='#ffbe7b',size=18);text(d,(780,y+185),'tight: '+tight['parsed']['color_class'],fill='#49fa8f',size=18)
 text(d,(20,937),'Manual tight crops are an oracle input diagnostic; all eight reference helmets are the same purple horse.',size=16)
 board.save(dest/'02_roi_vs_tight_cap.jpg',quality=95);manifest.append({'file':'highlights/02_roi_vs_tight_cap.jpg','claim':'Automatic broad rider-head ROI gives3/8purple; same classifier on posthoc manually tight crops gives8/8purple. Same purplehorse only; no11classaccuracy.'})
 # 3: Failed nearest association on new frames, with provisional same-horse link.
 rows=list(csv.DictReader((ROOT/'results/geometry_comparison.csv').open()));wrong=[r for r in rows if r['method']=='A_distance' and r['wrong']=='True'];board=Image.new('RGB',(1200,600),'#14202a');d=ImageDraw.Draw(board)
 for j,row in enumerate(wrong[:2]):
  f=ann[row['frame']];n=next(n for n in f['noses'] if n['id']==row['nose_id']);own=next(c for c in f['caps'] if c['id']==n['reference_cap_id']);bad=next(c for c in f['caps'] if c['display_number']==int(row['predicted_number']));points=[n['point'],own['center'],bad['center']];b=[max(0,int(min(p[0] for p in points)-60)),max(0,int(min(p[1] for p in points)-45)),min(1280,int(max(p[0] for p in points)+60)),min(610,int(max(p[1] for p in points)+65))];im=Image.open(ROOT/f['file']).crop(b);patch=contain(im,(540,420));sx=patch.width/(b[2]-b[0]);sy=patch.height/(b[3]-b[1]);ox=j*600+(600-patch.width)//2;oy=85;board.paste(patch,(ox,oy))
  xy=lambda p:(ox+(p[0]-b[0])*sx,oy+(p[1]-b[1])*sy)
  d.line([xy(n['point']),xy(own['center'])],fill='#49fa8f',width=4);d.line([xy(n['point']),xy(bad['center'])],fill='#ff5c60',width=4)
  for c in [own,bad]:text(d,xy(c['center']),f'cap{c["display_number"]}',size=20)
  text(d,(j*600+15,15),f'{f["pts"]:.0f}s nose of draft No.{n["reference_number"]}',size=23);text(d,(j*600+15,50),f'nearest picks {bad["display_number"]}; unique assignment picks {own["display_number"]}',size=18);text(d,(j*600+15,535),'GREEN: manual draft / RED: nearest rule',size=18)
 board.save(dest/'03_new_association_failures.jpg',quality=95);manifest.append({'file':'highlights/03_new_association_failures.jpg','claim':'Two new-frame nearest/gated errors; globaluniqueassignment resolves these under manual landmarks. References not user reviewed.'})
 # 4: Direct VLM localization picks a different horse on a clear side view.
 f=ann['t034'];im=Image.open(ROOT/f['file']).convert('RGB');d=ImageDraw.Draw(im)
 for suffix,col,label in [('_localize','#ff5c60','9B'),('_localize_27b','#ffcb42','27B')]:
  p=vlm['t034'+suffix]['parsed'];b=p['cap_box']
  if b:
   bx=[b[0]*1.28,b[1]*.72,b[2]*1.28,b[3]*.72];d.rectangle(bx,outline=col,width=4);text(d,(max(250,bx[0]-50),max(120,bx[1]-45)),label+' cap: '+p['cap_color'],fill=col,size=18)
 n=next(n for n in f['noses'] if n['reference_number']==9);c=next(c for c in f['caps'] if c['id']==n['reference_cap_id']);x,y=c['center'];d.ellipse((x-15,y-15,x+15,y+15),outline='#49fa8f',width=4);text(d,(x+20,y+20),'draft leader cap9',fill='#49fa8f',size=20);text(d,(250,20),'34s: full-frame localization examples (boxes are actual model outputs)',size=22)
 im.save(dest/'04_full_frame_localization.jpg',quality=95);manifest.append({'file':'highlights/04_full_frame_localization.jpg','claim':'Both testedVLMs localize a nonleader horse at34s under current prompt; image output is raw normalized coordinates mapped tosource.'})
 # 5: Motion direction and a valid rank coordinate are different problems.
 fid='t037';f=ann[fid];im=Image.open(ROOT/f['file']).convert('RGB');d=ImageDraw.Draw(im);axis=flow[fid]['direction_vector'];ranking=json.loads((ROOT/'results/rank_projection.json').read_text());row=next(r for r in ranking if r['frame']==fid and r['axis_source']=='flow_01_affine');selected=row['selected_nose_reference_number']
 for n in f['noses']:
  x,y=n['point'];col='#ff5c60' if n['reference_number']==selected else '#49fa8f';d.ellipse((x-7,y-7,x+7,y+7),outline=col,width=3);text(d,(x+12,y-10),f'draft No.{n["reference_number"]}',fill=col,size=19)
 arrow(d,(580,150),axis,100);text(d,(250,20),f'37s: automatic flow + screen projection chooses draft No.{selected}',size=22);text(d,(250,50),'Reference leader is9. A motion vector alone does not calibrate race position.',size=20)
 im.save(dest/'05_motion_is_not_rank_calibration.jpg',quality=95);manifest.append({'file':'highlights/05_motion_is_not_rank_calibration.jpg','claim':'Even mirror-consistent0.1sflow plus simple screen projection disagrees with provisional leader on3/8frames. Not a validated ground-motion/rank estimator.'})
 # 6: Cut candidate and exposure false alarm, full before/after pixels retained.
 board=Image.new('RGB',(1280,880),'#14202a');d=ImageDraw.Draw(board)
 for j,(fn,caption) in enumerate([('cut1.jpg','32.733s: visible race camera cut'),('cut3.jpg','44.933s: colour/exposure shift; same shot (visual review)')]):
  text(d,(15,j*440+8),caption,size=22);board.paste(Image.open(ROOT/'review/cuts'/fn),(0,j*440+40))
 board.save(dest/'06_cut_vs_colour_change.jpg',quality=95);manifest.append({'file':'highlights/06_cut_vs_colour_change.jpg','claim':'Histogram/MADcutcandidates require verification; one candidate is an exposure/colour shift rather than a camera cut.'})
 # 7: Survey-only input hazards, never used in the core8frame model comparisons.
 board=Image.new('RGB',(1280,840),'#14202a');d=ImageDraw.Draw(board)
 for j,(t,caption) in enumerate([(9,'9s: simultaneous camera views'),(18,'18s: floating broadcast identity labels')]):
  board.paste(Image.open(ROOT/'frames'/f't{t:06.2f}.jpg').resize((640,360)),(j*640,100));text(d,(j*640+12,50),caption,size=21)
 text(d,(20,490),'SURVEY ONLY — excluded from the core experiment inputs',fill='#ffcb42',size=26);text(d,(20,540),'A fixed corner/bottom mask does not remove every broadcast overlay.',size=23)
 board=board.crop((0,0,1280,610));board.save(dest/'07_input_hazards.jpg',quality=95);manifest.append({'file':'highlights/07_input_hazards.jpg','claim':'Compositeviews and floatingidentitylabels exist in source; these timestamps were excluded, not erased generatively.'})
 dump(dest/'manifest.json',manifest);print(json.dumps({'highlight_images':len(manifest)},indent=2))
if __name__=='__main__':main()
