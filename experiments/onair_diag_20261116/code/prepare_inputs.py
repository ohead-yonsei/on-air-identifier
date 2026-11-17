"""Timestamp-indexed frame extraction. Only video pixels enter the model."""
import bisect, hashlib, json, pathlib, subprocess, sys
from PIL import Image, ImageDraw
ROOT=pathlib.Path(__file__).resolve().parents[1]
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(args):
 r=subprocess.run(args,capture_output=True,text=True)
 if r.returncode: raise RuntimeError(r.stderr)
 return r

def prepare():
 cfg=json.loads((ROOT/'config.json').read_text());video=ROOT/cfg['video']
 probe=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)]).stdout)
 pts=[float(f['best_effort_timestamp_time']) for f in probe['frames']]
 assert pts==sorted(pts)
 dump(ROOT/'evidence/video_frame_pts.json',pts)
 specs=[]
 for sc in cfg['scenes']:
  start=sc['evaluation_video_seconds']-8; end=sc['evaluation_video_seconds']
  targets=[start+i/2 for i in range(16)]
  # Latest decoded frame no later than the target. No fps interpolation, no lookahead.
  inds=[bisect.bisect_right(pts,t+1e-8)-1 for t in targets]
  assert len(set(inds))==16 and min(inds)>=0
  assert all(start-1e-7<=pts[i]<=t+1e-7 and pts[i]<end for i,t in zip(inds,targets))
  d=ROOT/'frames'/sc['scene'];d.mkdir(parents=True,exist_ok=True)
  assert not list(d.glob('f*.jpg')), 'Refusing to overwrite existing frames'
  expr='+'.join(f'eq(n,{i})' for i in inds)
  vf=f"select='{expr}',scale=768:-2,drawbox=x=0:y=0:w=150:h=65:color=black:t=fill,drawbox=x=0:y=365:w=iw:h=ih-365:color=black:t=fill"
  cmd=['ffmpeg','-nostdin','-v','error','-i',str(video),'-map','0:v:0','-an','-sn','-dn','-to',str(end),'-vf',vf,'-fps_mode','vfr','-frames:v','16','-q:v','2',str(d/'f%03d.jpg')]
  run(cmd)
  files=sorted(d.glob('f*.jpg'));assert len(files)==16
  records=[]
  for k,(f,i,t) in enumerate(zip(files,inds,targets)):
   w,h=Image.open(f).size;assert (w,h)==(768,432)
   records.append({'file':str(f.relative_to(ROOT)),'sha256':sha(f),'decoded_frame_index':i,'target_video_seconds':t,'actual_pts_seconds':pts[i],'width':w,'height':h})
  spec={**sc,'video_file':cfg['video'],'source_sha256':sha(video),'source_time_origin':'container PTS in seconds; video first PTS 0.033333; no race-clock conversion used for scoring','window_start':start,'window_end_exclusive':end,'images_sent':16,'fps':2,'width':768,'audio_sent':False,'mask_rectangles_at_768':[[0,0,150,65],[0,365,768,67]],'ffmpeg_command':cmd,'frames':records}
  dump(d/'manifest.json',spec);specs.append(spec)
  sheet=Image.new('RGB',(768*4,464*4),'white');dr=ImageDraw.Draw(sheet)
  for k,f in enumerate(files):
   x=(k%4)*768;y=(k//4)*464;sheet.paste(Image.open(f),(x,y));dr.text((x+5,y+435),f"{sc['scene']} #{k+1}: PTS {pts[inds[k]]:.6f}s",fill='black')
  sheet.save(d/'contact.jpg',quality=90)
  # Review-only playback, reconstructed from precisely the images sent. No audio.
  run(['ffmpeg','-nostdin','-v','error','-framerate','2','-i',str(d/'f%03d.jpg'),'-an','-c:v','libx264','-pix_fmt','yuv420p',str(d/'input_review.mp4')])
 dump(ROOT/'scene_inputs.json',specs)
 print(json.dumps([{'scene':s['scene'],'n':16,'first':s['frames'][0]['actual_pts_seconds'],'last':s['frames'][-1]['actual_pts_seconds']} for s in specs]))
if __name__=='__main__': prepare()
