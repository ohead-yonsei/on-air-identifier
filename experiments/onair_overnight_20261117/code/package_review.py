"""Portable review ZIP and manifest. Does not package videos, weights or environments."""
import argparse,hashlib,json,re,zipfile
from pathlib import Path
from common import ROOT,dump,now,sha
MAC='/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261117/'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preliminary',action='store_true');a=ap.parse_args()
 # Complete local mirror manifest; raw videos/weights are verified separately on the server.
 skip={'videos','model_cache','ytdlp-lib','__pycache__','.git'}
 files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and not any(s in skip for s in p.relative_to(ROOT).parts) and p.name not in ['SHA256SUMS'] and p.suffix not in ['.zip','.tmp','.pyc']]
 (ROOT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
 name='REVIEW_PRELIMINARY.zip' if a.preliminary else 'ON_AIR_20261117_REVIEW.zip';target=ROOT/name
 prefixes=['highlights/','review/races/','results/','protocols/','code/','raw/human_mask']
 exact={'MORNING_REPORT_KO.md','README.md','PLAN_KO.md','config.json','annotations.json','frame_manifest.json','tracking_frame_manifest.json','SHA256SUMS','review/USER_REVIEW_KO.md','review/USER_REVIEW.csv','review/PREVIOUS_GEOMETRY_REVIEW.csv','review/LEADER_REVIEW.csv','review/QUESTIONS.csv','review/questions.json','review/leader.jpg','review/connections.jpg','raw/vlm.jsonl','raw/request_ledger.jsonl','raw/tracking.jsonl','raw/tracking_extra.jsonl','raw/tracking_flow.jsonl'}
 members=[]
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files+[ROOT/'SHA256SUMS']:
   rel=str(p.relative_to(ROOT))
   if rel not in exact and not rel.startswith(tuple(prefixes)) and not (rel.startswith('evidence/') and p.suffix in ['.json','.jsonl']):continue
   data=p.read_bytes()
   if p.suffix=='.md':
    text=data.decode();text=re.sub(r'(?<=\()'+re.escape(MAC)+r'([^\)]+)(?=\))',lambda m:str(Path(__import__('os').path.relpath(ROOT/m.group(1),p.parent))),text);data=text.encode()
   z.writestr(rel,data);members.append(rel)
  # Full source and input images intentionally remain in the experiment folder.
  z.writestr('ZIP_CONTENTS_KO.txt','이 ZIP은 검토 자료입니다. 전체 원시 입력 PNG, 원본 영상, 모델 가중치는 제외했습니다. SHA256SUMS는 전체 맥 동기화 폴더 기준이며 ZIP에 없는 파일도 포함합니다. ZIP의 Markdown은 상대경로로 변환하여 원본 MD와 해시가 다릅니다. 완전 재실행은 서버 실험 폴더와 README를 사용하세요.\n')
  z.writestr('ZIP_SHA256SUMS',''.join(hashlib.sha256(z.read(n)).hexdigest()+'  '+n+'\n' for n in z.namelist()))
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 print(name,'files',len(members),'bytes',target.stat().st_size,'sha256',sha(target))
if __name__=='__main__':main()
