"""Deterministic analytical crops; no generative image processing or model calls."""
from pathlib import Path
import json, hashlib
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
# Manually located on the original decoded 1280x720 frames, before predictions.
# x1,y1,x2,y2, with exclusive upper bounds. No graphics/saddlecloth digits included.
REGIONS = [
 ('A','t35',35.0,[190,412,216,439],[207,407,243,449],9),
 ('B','t35',35.0,[273,370,295,392],[277,374,320,417],1),
 ('C','t35',35.0,[263,348,284,370],[258,365,297,391],2),
 ('D','t35',35.0,[216,348,239,370],[218,357,261,391],10),
 ('E','t35',35.0,[462,337,484,359],[465,349,505,386],5),
 ('F','t35',35.0,[525,278,546,301],[524,288,567,324],11),
 ('G','t385',38.5,[691,466,725,499],[672,462,736,519],9),
 ('H','t385',38.5,[587,389,616,413],[573,402,633,448],1),
 ('I','t385',38.5,[475,344,501,370],[462,366,514,410],2),
 ('J','t385',38.5,[362,354,390,386],[355,378,408,415],10),
 ('K','t385',38.5,[682,330,707,356],[668,350,721,389],5),
 ('L','t385',38.5,[473,209,497,233],[455,228,511,266],11),
]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def font(n): return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', n)
def fit(im, size): return ImageOps.contain(im, size, Image.Resampling.LANCZOS)

def main():
    for d in ['crops','inputs','review']:(ROOT/d).mkdir(exist_ok=True)
    refs = json.loads((ROOT/'reference_manifest.json').read_text())
    board = Image.new('RGB',(640,540),'white');d = ImageDraw.Draw(board)
    for idx, r in enumerate(refs):
        im=Image.open(ROOT/r['file']).convert('RGBA')
        im=fit(im,(148,145));x=(idx%4)*160+(160-im.width)//2;y=(idx//4)*180+30
        board.paste(im,(x,y),im)
        d.text(((idx%4)*160+50,(idx//4)*180+5),f"No.{r['number']}",font=font(20),fill='black')
    board.save(ROOT/'references/registry.png')
    review=Image.new('RGB',(1056,1056),'#eeeeee');dr=ImageDraw.Draw(review)
    samples=[]
    for idx,(sid,fid,pts,capbox,torbox,label) in enumerate(REGIONS):
        frame=Image.open(ROOT/'source_frames'/f'{fid}.png').convert('RGB')
        assert frame.size==(1280,720)
        cap=frame.crop(capbox);torso=frame.crop(torbox)
        masked_caps=[]
        for other in REGIONS:
            if other[1]!=fid:continue
            b=other[3];inter=[max(b[0],torbox[0]),max(b[1],torbox[1]),min(b[2],torbox[2]),min(b[3],torbox[3])]
            if inter[0]<inter[2] and inter[1]<inter[3]:
                local=[inter[0]-torbox[0],inter[1]-torbox[1],inter[2]-torbox[0],inter[3]-torbox[1]]
                torso.paste((96,96,96),local);masked_caps.append({'sample_id':other[0],'source_box':inter})
        cp=ROOT/'crops'/f'{sid}_cap.png';tp=ROOT/'crops'/f'{sid}_torso.png'
        cap.save(cp);torso.save(tp)
        parts={'cap':cap,'torso':torso}
        input_meta={}
        for condition,enabled in [('H',['cap']),('S',['torso']),('HS',['cap','torso'])]:
            canvas=Image.new('RGB',(336,224),(96,96,96))
            for name,pos in [('cap',(16,48)),('torso',(184,48))]:
                if name in enabled:
                    patch=fit(parts[name],(136,128))
                    canvas.paste(patch,(pos[0]+(136-patch.width)//2,pos[1]+(128-patch.height)//2))
            p=ROOT/'inputs'/f'{sid}_{condition}.png';canvas.save(p)
            input_meta[condition]={'file':str(p.relative_to(ROOT)),'sha256':sha(p),'size':[336,224]}
        # Review-only label includes ID/time and source pixel sizes, never target number.
        x=(idx%3)*352;y=(idx//3)*264
        dr.text((x+10,y+6),f'{sid}  {pts:.1f}s  cap {cap.width}x{cap.height}',font=font(19),fill='black')
        review.paste(Image.open(ROOT/input_meta['HS']['file']),(x+8,y+32))
        samples.append({'id':sid,'frame_id':fid,'pts_seconds':pts,'source_file':f'source_frames/{fid}.png','source_sha256':sha(ROOT/'source_frames'/f'{fid}.png'),'cap_box':capbox,'torso_box':torbox,'torso_masked_helmet_regions':masked_caps,'cap_file':str(cp.relative_to(ROOT)),'torso_file':str(tp.relative_to(ROOT)),'provisional_number':label,'label_status':'assistant visual label; user review requested','cap_native_size':list(cap.size),'torso_native_size':list(torso.size),'inputs':input_meta})
    review.save(ROOT/'review/crop_review.jpg',quality=95)
    (ROOT/'samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
    plan={'conditions':{'H':'helmet only','S':'silks only','HS':'same helmet pixels plus same silks pixels'},'sample_count':12,'source_frames':[35.0,38.5],'candidate_numbers':list(range(1,12)),'max_vlm_calls':36,'repeat_inference':False,'model':'qwen3.5:9b','model_digest':'6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7','options':{'temperature':0,'seed':20261116,'num_ctx':16384,'num_predict':96,'top_k':20,'top_p':1,'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0},'endpoint':'http://127.0.0.1:11435','query_canvas':[336,224],'reference_registry':[640,540],'cap_pixels_identical_between_H_and_HS':True,'torso_pixels_identical_between_S_and_HS':True,'scope':'Manual ROI identification, not detector/tracker/order accuracy. No trained classifier. All model calls independent. No ranked race outcomes or target labels in input.','reference_note':'User-provided 11 silk images mapped via official race owner identities. Historical results fields discarded. Verify against dated racecard and actual imagery.','timing_note':'Identical canvas size and same reference image across conditions controls visible information. This does not estimate savings from a smaller native-resolution deployment.'}
    (ROOT/'config.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'samples':len(samples),'input_images':len(samples)*3,'review':str(ROOT/'review/crop_review.jpg')}))

if __name__=='__main__': main()
