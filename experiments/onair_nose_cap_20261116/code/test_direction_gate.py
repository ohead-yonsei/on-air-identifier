"""Post-hoc direction-gating probe, explicitly not held-out validation."""
from pathlib import Path
import csv,json,time,hashlib
import numpy as np
from PIL import Image,ImageDraw
from run_association import ROOT,ANN,font,label,mark

CONFIG={'status':'post-hoc: motivated by four nearest-cap failures on the same 11 annotations; no held-out sample','rule':'Cap must lie at or behind nose along manually supplied screen progression axis: dot(cap-nose,axis)<=0. Among eligible caps choose nearest Euclidean distance. Abstain if none.','axis_offsets_degrees':[-30,-15,0,15,30],'seed':20261116,'coordinate_jitter_repeats':2000,'nose_jitter_xy_pixels':5,'cap_jitter_xy_pixels':2,'timing_repeats':1000,'new_model_calls':0}

def gated(N,C,D):
    axis=D/np.linalg.norm(D)
    delta=C[None,:,:]-N[:,None,:]
    cost=np.linalg.norm(delta,axis=2)
    cost[(delta@axis)>0]=np.inf
    pred=cost.argmin(axis=1)
    pred[np.isinf(cost.min(axis=1))]=-1
    return pred

def rotate(D,deg):
    a=np.deg2rad(deg)
    return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@D

def main():
    out=ROOT/'posthoc_direction';out.mkdir(exist_ok=True)
    (out/'config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
    rng=np.random.default_rng(CONFIG['seed']);rows=[];summaries=[];raw=[];timing=[];boards=[]
    for fr in ANN['frames']:
        caps=fr['caps'];noses=fr['noses'];D=np.array(fr['direction'],float)
        co=rng.permutation(len(caps));no=rng.permutation(len(noses));caps=[caps[i] for i in co];noses=[noses[i] for i in no]
        C=np.array([c['center'] for c in caps]);N=np.array([n['point'] for n in noses]);target=np.array([next(i for i,c in enumerate(caps) if c['id']==n['reference_cap_id']) for n in noses])
        for deg in CONFIG['axis_offsets_degrees']:
            axis=rotate(D,deg);pred=gated(N,C,axis)
            summaries.append({'frame':fr['id'],'axis_offset_degrees':deg,'matches_provisional_reference':int(np.sum(pred==target)),'n':len(N),'abstentions':int(np.sum(pred<0))})
            for i,n in enumerate(noses):
                rows.append({'frame':fr['id'],'axis_offset_degrees':deg,'reference_number':n['reference_number'],'predicted_number':None if pred[i]<0 else caps[pred[i]]['display_number'],'matches_provisional_reference':bool(pred[i]==target[i])})
            raw.append({'frame':fr['id'],'axis_offset_degrees':deg,'manual_axis':axis.tolist(),'input_nose_coordinates':N.tolist(),'input_cap_coordinates':C.tolist(),'predicted_cap_indices':pred.tolist(),'nose_ids_outside_solver':[n['id'] for n in noses],'cap_ids_outside_solver':[c['id'] for c in caps]})
        ts=[]
        for _ in range(20):gated(N,C,D)
        for _ in range(CONFIG['timing_repeats']):
            s=time.perf_counter_ns();gated(N,C,D);ts.append((time.perf_counter_ns()-s)/1e6)
        match=np.zeros(len(N),int)
        for _ in range(CONFIG['coordinate_jitter_repeats']):
            p=gated(N+rng.uniform(-5,5,N.shape),C+rng.uniform(-2,2,C.shape),D);match+=(p==target)
        timing.append({'frame':fr['id'],'median_ms':float(np.median(ts)),'p95_ms':float(np.percentile(ts,95)),'scope':'coordinate geometry only; no detection/image decode/cap recognition','jitter_match_by_reference_number':{str(n['reference_number']):float(match[i]/CONFIG['coordinate_jitter_repeats']) for i,n in enumerate(noses)}})
        raw.append({'frame':fr['id'],'runtime_timings_ms':ts})
        # Illustrates two distinct lines, not a calibrated equal-progress contour.
        src=Image.open(ROOT/'inputs'/(fr['id']+'.png')).convert('RGB')
        crop=(130,310,620,590) if fr['id']=='t35' else (300,280,790,610)
        im=src.crop(crop);scale=1.35;im=im.resize((round(im.width*scale),round(im.height*scale)))
        board=Image.new('RGB',(720,660),'#17222c');board.paste(im,(30,115));draw=ImageDraw.Draw(board)
        def xy(p):return np.array([30+(p[0]-crop[0])*scale,115+(p[1]-crop[1])*scale])
        axis=D/np.linalg.norm(D);perp=np.array([-axis[1],axis[0]])
        for num in [9,5]:
            n=next(n for n in noses if n['reference_number']==num); c=next(c for c in caps if c['id']==n['reference_cap_id']);p=xy(n['point']);q=xy(c['center'])
            draw.line([tuple(p-perp*100),tuple(p+perp*100)],fill='#ffce3a',width=3)
            draw.line([tuple(p),tuple(q)],fill='#32ff73',width=3);mark(draw,p,'#ffce3a')
            label(draw,tuple(q+[10,-15]),f'cap {num}',size=19)
        a=np.array([530.,155.]);b=a+axis*85
        draw.line([tuple(a),tuple(b)],fill='#54d4ff',width=5)
        draw.polygon([tuple(b),tuple(b-axis*17+perp*9),tuple(b-axis*17-perp*9)],fill='#54d4ff')
        label(draw,(20,14),f'{fr["pts"]:.1f}s | one MANUAL axis for this view',size=22)
        label(draw,(20,48),'CYAN arrow = approximate progression',color='#54d4ff',size=20)
        label(draw,(20,79),'YELLOW = perpendicular line through each nose',color='#ffce3a',size=20)
        label(draw,(20,585),'GREEN = same-horse nose-to-cap reference link',color='#32ff73',size=20)
        label(draw,(20,620),'No automatic angle estimation or camera calibration.',size=19)
        boards.append(board)
    hero=Image.new('RGB',(1440,660));hero.paste(boards[0],(0,0));hero.paste(boards[1],(720,0));hero.save(ROOT/'review'/'direction_vs_connection.png')
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in raw))
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (out/'timing.json').write_text(json.dumps(timing,indent=2)+'\n')
    (out/'run.json').write_text(json.dumps({'runtime_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':CONFIG['status']},indent=2)+'\n')
    print(json.dumps({'summaries':summaries,'timing':timing},indent=2))

if __name__=='__main__':main()
