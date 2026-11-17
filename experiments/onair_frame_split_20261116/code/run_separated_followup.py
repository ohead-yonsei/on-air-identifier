"""Prepared follow-up; NOT used for the completed 16 calls. Fresh output directory required.
Example: python code/run_separated_followup.py /absolute/path/to/new_run --write-prompts
To explicitly run later, replace --write-prompts with --max-new 1, inspect, then --max-new 15.
"""
import json,pathlib,shutil,sys
import run_diagnostic as base
source=base.ROOT
def separated_body(sc,c,cfg):
 body,manifest=original_build(sc,c,cfg)
 if c!='F-no-video':
  markers='\n'.join(f'Frame {i+1:02d} at relative {i/2:.1f}s: [img]' for i in range(16))
  body['messages'][1]['content']+='\n\nCHRONOLOGICAL FRAMES\n'+markers
  manifest['messages'][1]['content']=body['messages'][1]['content']
  manifest['image_layout']='explicit placeholders separated by frame labels'
 return body,manifest
original_build=base.build
if __name__=='__main__':
 if len(sys.argv)<2:raise SystemExit(__doc__)
 dest=pathlib.Path(sys.argv.pop(1)).resolve()
 if dest==source:raise SystemExit('A new directory is required to preserve the first 16 calls.')
 if not dest.exists():
  dest.mkdir(parents=True)
  for n in ['config.json','scene_inputs.json']:shutil.copy2(source/n,dest/n)
  (dest/'frames').symlink_to(source/'frames',target_is_directory=True)
 base.ROOT=dest;base.build=separated_body;base.main()
