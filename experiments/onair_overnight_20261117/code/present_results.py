"""Render evidence and review material from immutable saved outputs; no inference."""
import csv,json,math,shutil,statistics
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
from common import ROOT,CFG,dump,readlines,now

MAC=Path('/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n=20):return ImageFont.truetype(FONT,n)
def fit(im,size):return ImageOps.contain(im,size,Image.Resampling.LANCZOS)
def label(im,xy,s,n=20,colour='white'):ImageDraw.Draw(im).text(xy,s,font=font(n),fill=colour)
def link(p):return str(MAC/p)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def write_csv(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 ann=json.loads((ROOT/'annotations.json').read_text())['samples'];samples={s['id']:s for s in ann};fm={f['id']:f for f in json.loads((ROOT/'frame_manifest.json').read_text())}
 rows=[r for p in (ROOT/'results').glob('regions_*.json') for r in json.loads(p.read_text())];by={(r['sample_id'],r['method']):r for r in rows};raw={r['item_id']:r for r in readlines(ROOT/'raw/vlm.jsonl')};methods=CFG['methods'];hl=ROOT/'highlights';hl.mkdir(exist_ok=True)
 if (ROOT/'results/human_mask_control.json').exists():
  for r in json.loads((ROOT/'results/human_mask_control.json').read_text()):
   by[(r['sample_id'],r['method'])]=r
   if r['vlm'].get('reused_item_id'):raw[r['item_id']]=raw[r['vlm']['reused_item_id']]
 cases=[('01_yellow','R04_t050_s2','노랑 초안: 넓은 영역의 빨강 판정이 수동 영역에서 노랑으로 바뀜'),('02_green','R01_t030_s3','초록 초안: 넓은 영역의 파랑 판정이 수동 영역에서 초록으로 바뀜'),('03_human_sky','HUMAN_J','사람 확인 10번: 마스킹 보정 후에도 파랑 5번으로 혼동'),('04_human_stripes','HUMAN_L','사람 확인 11번: 마스킹 보정에서도 수동 영역은 1번으로 악화'),('05_unresolved','R08_t080_s1','초안도 미확인: 모델의 번호를 정답으로 삼지 않음')]
 manifests=[]
 for stem,sid,caption in cases:
  s=samples[sid];src=Image.open(ROOT/fm[s['frame']]['file']).convert('RGB');can=Image.new('RGB',(1480,620),'#18202a');ref=str(s['number']) if s['number'] else 'unresolved'
  label(can,(20,12),f"{sid} | reference={ref} | {s['label_status']}",24)
  if sid.startswith('HUMAN_'):label(can,(20,45),'Post-result masked input correction; byte-identical crop responses may be reused.',17,'#ffdf7e')
  l,t,r,b=s['person_box'];pad=20;ctx=src.crop((max(0,int(l)-pad),max(0,int(t)-pad),min(src.width,int(r)+pad),min(src.height,int(b)+pad)));ctx=fit(ctx,(240,360));can.paste(ctx,(20,100));label(can,(20,480),'Common rider context',18);label(can,(20,510),'No model output',16)
  for j,m in enumerate(methods):
   q=by[(sid,m)];x=280+j*238;label(can,(x,65),m,20)
   if q.get('image'):can.paste(Image.open(ROOT/q['image']).convert('RGB'),(x,108))
   else:label(can,(x,185),'NO ROI / no request',16)
   pred=raw.get(q['item_id'],{}).get('parsed',{}).get('color_class','not_called');label(can,(x,350),'Qwen JSON color_class:',14);label(can,(x,376),pred,14,'#ffdf7e');label(can,(x,409),f"mapped no.={q['vlm']['number'] or 'unknown'}",18)
   label(can,(x,444),f"HSV no.={q['hsv']['number'] or 'unknown'}",18);label(can,(x,472),f"cap IoU={q['cap_iou']:.3f}",18);label(can,(x,500),f"native={q.get('native_size','none')}",16)
  label(can,(20,570),'Saved model inputs shown exactly at 224 x 224; labels are outside input pixels. U is manual, not an automatic detector.',17)
  path=hl/(stem+'.jpg');can.save(path,quality=96);manifests.append({'file':str(path.relative_to(ROOT)),'samples':[sid],'caption':caption,'reference_status':s['label_status'],'outputs':[{'item':by[(sid,m)]['item_id'],'raw':raw.get(by[(sid,m)]['item_id'],{}).get('parsed'),'number':by[(sid,m)]['vlm']['number']} for m in methods]})
 # Each race is one review image and one question. Predictions are deliberately omitted.
 review=ROOT/'review';(review/'races').mkdir(parents=True,exist_ok=True);questions=[];colour=['unresolved','white','yellow','red','black','blue','green','brown/maroon','pink','purple','sky blue','sky/white stripes'];review_rows=[]
 priority=['R01','R02','R04','R08','R09','R10','R03','R05','R06','R07']
 for k,race in enumerate(priority,1):
  ss=[s for s in ann if s['race']==race];im=Image.new('RGB',(1260,900),'#18202a');label(im,(20,12),f'{race} | DRAFT labels only | Read left to right, top to bottom',26)
  for i,s in enumerate(ss):
   src=Image.open(ROOT/fm[s['frame']]['file']).convert('RGB');l,t,r,b=s['person_box'];l=max(0,int(l)-12);t=max(0,int(t)-12);r=min(src.width,int(r)+12);b=min(src.height,int(b)+12)
   crop=src.crop((l,t,r,b));dr=ImageDraw.Draw(crop);box=s['cap_box'];dr.rectangle([box[0]-l,box[1]-t,box[2]-l,box[3]-t],outline='#51ff91',width=2);patch=fit(crop,(380,310));x=20+(i%3)*415;y=65+(i//3)*415;im.paste(patch,(x+(380-patch.width)//2,y));label(im,(x,y+320),f"Photo {i+1} | {fm[s['frame']]['actual_pts']:g}s | no. {s['number'] or '?'}",22);label(im,(x,y+351),colour[s['number'] or 0],20,'#ffdf7e')
   review_rows.append({'question_id':f'Q{k:02}','sample_id':s['id'],'race':race,'photo':i+1,'pts':fm[s['frame']]['actual_pts'],'draft_number':s['number'] or 'unclear','draft_colour':colour[s['number'] or 0],'image':f'review/races/{race}.jpg','label_scope':'assistant_draft_colour; box is not confirmed','user_confirmed':'','corrected_number':'','comment':''})
  path=review/'races'/(race+'.jpg');im.save(path,quality=95);questions.append({'id':f'Q{k:02}','image':str(path.relative_to(ROOT)),'question':f'{race}: 사진 아래의 모자 번호가 맞나요? 틀린 사진 번호와 올바른 마번만 적고, 애매하면 미확인으로 남겨 주세요.','seconds':40})
 old=Path(CFG['old_experiment']);geo=json.loads((old/'geometry_annotations.json').read_text())['frames'];tracking={round(r['pts'],1):r for r in readlines(ROOT/'raw/tracking.jsonl')};ts=json.loads((ROOT/'results/tracking_summary.json').read_text());ref={}
 for r in ts['references']:ref[(r['pts'],r['number_draft'])]=r
 for kind in ['leader','connections','tracks']:
  panel=Image.new('RGB',(1600,1660),'#18202a');label(panel,(20,12),{'leader':'Provisional leader: no.9? Please confirm each time.','connections':'Are the green nose-to-cap links on the same horse/rider?','tracks':'Saved tracking outputs | IDs are NOT horse numbers'}[kind],27)
  for i,g in enumerate(geo):
   t=g['pts'];rec=tracking[round(t,1)];src=Image.open(ROOT/'tracking/frames'/f'track_{round(t*10):04}.png').convert('RGB');dr=ImageDraw.Draw(src)
   for n in g['noses']:
    cap=next(c['center'] for c in g['caps'] if c['id']==n['reference_cap_id']);p=n['point'];num=n['reference_number']
    if kind=='leader' and num!=9:continue
    if kind=='connections':dr.line([tuple(p),tuple(cap)],fill='#4eff90',width=4);dr.ellipse([p[0]-6,p[1]-6,p[0]+6,p[1]+6],outline='#ffdf7e',width=3)
    dr.ellipse([cap[0]-10,cap[1]-10,cap[0]+10,cap[1]+10],outline='#4eff90',width=3)
    txt=f'draft {num}'
    tx=min(cap[0]+12,src.width-100)
    dr.text((tx,cap[1]-10),txt,font=font(17),fill='white',stroke_fill='black',stroke_width=2)
   x=(i%2)*800;y=62+(i//2)*398;patch=fit(src,(780,340));panel.paste(patch,(x+(800-patch.width)//2,y))
   caption=f"{t:g}s | shot={rec['trackers']['affine']['shot']} | view={rec['view_proxy']}" if kind=='tracks' else f"{t:g} seconds | Provisional leader: no.9?" if kind=='leader' else f"{t:g} seconds | DRAFT nose-to-cap links"
   label(panel,(x+15,y+345),caption,17)
   if kind=='tracks':
    labels=[]
    for n in g['noses']:
     rr=ref[(t,n['reference_number'])];labels.append(f"{n['reference_number']}: raw {rr['tracks']['no_warp']['person_track']}/aff {rr['tracks']['affine']['person_track']}")
    label(panel,(x+15,y+370),'  |  '.join(labels),16)
  path=(hl/'06_tracking.jpg') if kind=='tracks' else (review/(kind+'.jpg'));panel.save(path,quality=95)
  if kind=='tracks':manifests.append({'file':str(path.relative_to(ROOT)),'caption':'동일 화면 안에서 사람 트랙 ID가 바뀌며, 박스 비율 뷰 추정은 측면을 놓침','reference_status':'assistant geometry drafts, not MOT ground truth','model_outputs':'person track IDs from raw/tracking.jsonl; draft numbers used only for annotation'})
  else:questions.append({'id':'Q11' if kind=='leader' else 'Q12','image':str(path.relative_to(ROOT)),'question':'각 시점의 선두가 표시된 보라색 9번인가요? 다른 시점·미확인 시점만 알려 주세요.' if kind=='leader' else '녹색 선이 같은 말의 코끝과 그 말의 기수 모자를 잇고 있나요? 틀리거나 애매한 시점·번호만 알려 주세요.','seconds':60 if kind=='leader' else 120})
 shutil.copy2(old/'review/USER_REVIEW.csv',review/'PREVIOUS_GEOMETRY_REVIEW.csv')
 write_csv(review/'USER_REVIEW.csv',review_rows)
 write_csv(review/'LEADER_REVIEW.csv',[{'pts':g['pts'],'draft_leader':9,'user_confirmed':'','corrected_leader':'','comment':''} for g in geo])
 write_csv(review/'QUESTIONS.csv',[{**q,'user_confirmed':'','comment':''} for q in questions]);dump(review/'questions.json',questions)
 review_md='# 10분 검수\n\nQ1~Q10은 경주마다 사진 여섯 장입니다. **사진 아래 번호가 맞는지**만 봐 주세요. 예: “R2 사진 3은 5번, 나머지는 맞음”. 애매하면 미확인입니다. 모자 박스 위치·코끝 좌표·전체 순위를 동시에 승인한 것으로 처리하지 않습니다.\n\n경주별 40초×10 + 선두 60초 + 연결 120초 = 약 9분 40초입니다. 우선 평가 경주 여섯 장부터 확인해도 됩니다. 모든 확인 칸은 비어 있습니다.\n\n'
 for q in questions:review_md+=f"## {q['id']} · 약 {q['seconds']}초\n\n{q['question']}\n\n![{q['id']}]({link(q['image'])})\n\n"
 review_md+='번호 초안 60개는 USER_REVIEW.csv, 어제 코끝–모자 22개는 PREVIOUS_GEOMETRY_REVIEW.csv, 선두 8시점은 LEADER_REVIEW.csv에 따로 기록합니다. 10월 38.5초의 사람 확인 6개는 재확인 대상과 합치지 않았습니다.\n'
 (review/'USER_REVIEW_KO.md').write_text(review_md)
 # Optional seventh image only after the predeclared extra experiment completes.
 if (ROOT/'results/resolution_extra.json').exists():
  extra=json.loads((ROOT/'results/resolution_extra.json').read_text());chosen=extra[0]['sample_id'];rr=[x for x in extra if x['sample_id']==chosen];rr.append(by[(chosen,'U_manual')]);im=Image.new('RGB',(990,430),'#18202a');label(im,(20,15),f'{chosen} | first predeclared resolution sample',24)
  for j,r in enumerate(rr):
   x=20+j*325;im.paste(Image.open(ROOT/r['image']),(x,75));out=raw[r['item_id']]['parsed']['color_class'];label(im,(x,315),f"source width {r.get('width',1920)}",20);label(im,(x,346),out,18);label(im,(x,377),f"native cap={r.get('native_cap_size',r.get('native_size'))}",17)
  path=hl/'07_resolution.jpg';im.save(path,quality=96);manifests.append({'file':str(path.relative_to(ROOT)),'caption':'사전 지정 첫 표본: 축소 후 모자 픽셀과 실제 출력 비교','sample':chosen,'selection':'first preregistered sample, not best/worst outcome'})
 dump(hl/'manifest.json',manifests);index='# 의미 있는 사례\n\n모델에 넣은 입력에는 여기 표시한 번호·정답·출력이 없습니다. 성공·실패 사례는 결과를 본 뒤 설명 목적으로 선택했으며 표본 평균을 대신하지 않습니다.\n\n'
 for i,m in enumerate(manifests,1):index+=f"## {i}. {m['caption']}\n\n근거 범위: `{m['reference_status'] if 'reference_status' in m else 'assistant draft'}`.\n\n![사례 {i}]({link(m['file'])})\n\n"
 (hl/'INDEX_KO.md').write_text(index)
 dump(ROOT/'evidence/presentation_manifest.json',{'created_utc':now(),'highlights':len(manifests),'review_questions':len(questions),'review_draft_caps':len(review_rows),'prior_geometry_pending':22,'no_confirmation_written':True})
 print('Rendered',len(manifests),'highlights and',len(questions),'review questions')
if __name__=='__main__':main()
