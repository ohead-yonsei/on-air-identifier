"""Manual-landmark geometry diagnostic. Never calls a VLM or a detector."""
from pathlib import Path
import csv, hashlib, json, platform, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config.json').read_text())
ANN = json.loads((ROOT / 'annotations.json').read_text())
METHODS = list(CFG['methods'])

def font(n):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', n)

def predict(noses, caps, direction, method):
    # Only arrays of coordinates enter the algorithms. No number, reference link,
    # original IDs, color, silks, results graphic, or leaderboard is available.
    delta = noses[:, None, :] - caps[None, :, :]
    if method == 'line_nearest':
        axis = direction / np.linalg.norm(direction)
        cost = np.abs(delta @ axis)
        return cost.argmin(axis=1), cost
    cost = np.linalg.norm(delta, axis=2)
    if method == 'point_nearest':
        return cost.argmin(axis=1), cost
    rows, cols = linear_sum_assignment(cost)
    out = np.empty(len(noses), dtype=int)
    out[rows] = cols
    return out, cost

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def label(draw, xy, text, color='white', size=18):
    f=font(size); box=draw.textbbox(xy,text,font=f)
    draw.rectangle((box[0]-3,box[1]-2,box[2]+3,box[3]+2),fill='#17222c')
    draw.text(xy,text,font=f,fill=color)

def mark(draw, p, color):
    x,y=p; draw.line((x-5,y-5,x+5,y+5),fill=color,width=3)
    draw.line((x-5,y+5,x+5,y-5),fill=color,width=3)

def main():
    for d in ['inputs','review','results','evidence']:(ROOT/d).mkdir(exist_ok=True)
    rng=np.random.default_rng(CFG['seed'])
    all_rows=[]; raw=[]; summaries=[]; timing=[]; checks=[]
    panel_parts=[]
    for fr in ANN['frames']:
        fid=fr['id']; caps=fr['caps']; noses=fr['noses']
        src=Image.open(ROOT/'source_frames'/f'{fid}.png').convert('RGB')
        assert src.size==(1280,720)
        masked=src.copy()
        for b in CFG['graphics_masks']:masked.paste((48,48,48),b)
        masked.save(ROOT/'inputs'/f'{fid}.png')
        checks.append({'name':fid+'_graphics_masks','passed':all(np.all(np.asarray(masked)[b[1]:b[3],b[0]:b[2]]==48) for b in CFG['graphics_masks'])})
        # Shuffling ensures that original annotation ordering cannot be the solver.
        c_order=rng.permutation(len(caps)); n_order=rng.permutation(len(noses))
        cs=[caps[i] for i in c_order]; ns=[noses[i] for i in n_order]
        C=np.array([c['center'] for c in cs],float)
        N=np.array([n['point'] for n in ns],float)
        D=np.array(fr['direction'],float)
        lookup={c['id']:j for j,c in enumerate(cs)}
        target=np.array([lookup[n['reference_cap_id']] for n in ns])
        fn_by_id={}; predictions={}
        for method in METHODS:
            pred,cost=predict(N,C,D,method); predictions[method]=pred
            # Independent row/column shuffle invariance (ties may be nonunique).
            qn=rng.permutation(len(N)); qc=rng.permutation(len(C))
            rp,rc=predict(N[qn],C[qc],D,method)
            mapped=np.empty(len(N),int); mapped[qn]=qc[rp]
            checks.append({'name':fid+'_'+method+'_permutation_invariant','passed':bool(np.array_equal(pred,mapped))})
            if method=='point_one_to_one':assert len(set(pred.tolist()))==len(pred)
            ts=[]
            for _ in range(20):predict(N,C,D,method)
            for _ in range(CFG['timing_repeats']):
                start=time.perf_counter_ns(); predict(N,C,D,method); ts.append((time.perf_counter_ns()-start)/1e6)
            timing.append({'frame':fid,'method':method,'queries':len(N),'candidate_caps':len(C),'repeats':len(ts),'median_ms':float(np.median(ts)),'p95_ms':float(np.percentile(ts,95)),'scope':CFG['timing_includes']})
            jitter_match=np.zeros(len(N),int); jitter_same=np.zeros(len(N),int)
            for _ in range(CFG['sensitivity']['repeats']):
                nj=N+rng.uniform(-5,5,N.shape); cj=C+rng.uniform(-2,2,C.shape)
                jp,_=predict(nj,cj,D,method)
                jitter_match+=(jp==target); jitter_same+=(jp==pred)
            for i,n in enumerate(ns):
                own=target[i]; chosen=pred[i]
                row={'frame':fid,'pts_seconds':fr['pts'],'nose_id':n['id'],'reference_number':n['reference_number'],'reference_cap_id':n['reference_cap_id'],'method':method,'predicted_cap_id':cs[chosen]['id'],'predicted_number':cs[chosen]['display_number'],'matches_provisional_reference':bool(chosen==own),'nose_x':N[i,0],'nose_y':N[i,1],'chosen_cost_px':float(cost[i,chosen]),'own_cap_cost_px':float(cost[i,own]),'cost_difference_own_minus_chosen_px':float(cost[i,own]-cost[i,chosen]),'jitter_reference_match_fraction':float(jitter_match[i]/CFG['sensitivity']['repeats']),'jitter_same_prediction_fraction':float(jitter_same[i]/CFG['sensitivity']['repeats']),'reference_status':n['reference_status'],'cap_number_status':cs[own]['number_status']}
                all_rows.append(row); fn_by_id[(n['id'],method)]=row
            summaries.append({'frame':fid,'method':method,'reference_agreements':int(np.sum(pred==target)),'reference_disagreements':int(np.sum(pred!=target)),'scored_noses':len(N),'candidate_caps':len(C),'status':'assistant provisional association reference, not expert accuracy'})
            raw.append({'frame':fid,'method':method,'input_nose_coordinates':N.tolist(),'input_cap_coordinates':C.tolist(),'manual_direction':D.tolist(),'nose_ids_outside_solver':[n['id'] for n in ns],'cap_ids_outside_solver':[c['id'] for c in cs],'predicted_cap_indices':pred.tolist(),'cost_matrix':cost.tolist(),'runtime_timings_ms':ts})
        # Reference review: every link is a manually traced association, not output.
        ref=masked.copy(); dr=ImageDraw.Draw(ref)
        for n in noses:
            c=caps[next(i for i,c in enumerate(caps) if c['id']==n['reference_cap_id'])]
            dr.line([tuple(n['point']),tuple(c['center'])],fill='#32ff73',width=3)
            mark(dr,n['point'],'#ffce3a');dr.rectangle(c['box'],outline='#32ff73',width=2)
            label(dr,(c['box'][2]+3,c['box'][1]-12),str(n['reference_number']),size=20)
        label(dr,(250,20),f'{fr["pts"]:.1f}s | DRAFT same-horse links | green=reference, X=nose',size=20)
        ref.save(ROOT/'review'/f'{fid}_reference_links.png')
        # Full comparison: red lines only disagreements, green reference lines.
        comp=ref.copy(); dc=ImageDraw.Draw(comp)
        for n in noses:
            row=fn_by_id[(n['id'],'point_nearest')]
            if not row['matches_provisional_reference']:
                cp=next(c for c in caps if c['id']==row['predicted_cap_id'])
                dc.line([tuple(n['point']),tuple(cp['center'])],fill='#ff414b',width=3)
        label(dc,(250,52),'RED = closest cap when it disagrees with same-horse draft',color='#ff7b81',size=18)
        comp.save(ROOT/'review'/f'{fid}_nearest_comparison.png')
        example_num=11 if fid=='t35' else 5
        n=next(n for n in noses if n['reference_number']==example_num)
        row=fn_by_id[(n['id'],'point_nearest')]
        own=next(c for c in caps if c['id']==n['reference_cap_id'])
        wrong=next(c for c in caps if c['id']==row['predicted_cap_id'])
        crop=(400,260,620,445) if fid=='t35' else (620,310,755,595)
        scale=2 if fid=='t35' else 1.65
        im=masked.crop(crop).resize((round((crop[2]-crop[0])*scale),round((crop[3]-crop[1])*scale)),Image.Resampling.NEAREST)
        board=Image.new('RGB',(600,650),'#17222c');board.paste(im,((600-im.width)//2,100))
        db=ImageDraw.Draw(board)
        ox=(600-im.width)//2
        def xy(p):return (ox+(p[0]-crop[0])*scale,100+(p[1]-crop[1])*scale)
        db.line([xy(n['point']),xy(own['center'])],fill='#32ff73',width=4)
        db.line([xy(n['point']),xy(wrong['center'])],fill='#ff414b',width=4)
        mark(db,xy(n['point']),'#ffce3a')
        for c in [own,wrong]:
            p=xy(c['center']);label(db,(p[0]+12,p[1]-26),f'cap {c["display_number"]}',size=20)
        label(db,(20,16),f'{fr["pts"]:.1f}s | nose of No.{example_num}',size=23)
        label(db,(20,51),f'Nearest cap: No.{row["predicted_number"]} ({row["chosen_cost_px"]:.1f}px)',color='#ff7b81',size=21)
        label(db,(20,590),f'Same horse: No.{example_num} ({row["own_cap_cost_px"]:.1f}px)',color='#32ff73',size=21)
        label(db,(20,623),'Reference link is visually annotated, not a model.',size=17)
        panel_parts.append(board)
        # Leader projection limited to annotated candidate noses; never field rank.
        projected=N@(D/np.linalg.norm(D)); leader=ns[int(projected.argmax())]
        raw.append({'frame':fid,'operation':'leader_screen_projection','manual_direction':D.tolist(),'candidate_count':len(N),'nose_projections_px':dict(zip([n['id'] for n in ns],projected.tolist())),'selected_reference_number':leader['reference_number'],'limitation':'Only annotated candidates and manual screen axis; no full-field rank or calibrated distance.'})
    hero=Image.new('RGB',(1200,650));hero.paste(panel_parts[0],(0,0));hero.paste(panel_parts[1],(600,0));hero.save(ROOT/'review'/'nearest_failure_examples.png')
    with (ROOT/'results'/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    (ROOT/'results'/'raw.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in raw))
    (ROOT/'results'/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (ROOT/'results'/'timing.json').write_text(json.dumps(timing,indent=2)+'\n')
    (ROOT/'evidence'/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    import scipy, PIL
    environment={'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pillow':PIL.__version__,'execution_location':str(ROOT),'GPU_used':False,'new_model_calls':0,'runtime_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'config_sha256':sha(ROOT/'config.json'),'annotations_sha256':sha(ROOT/'annotations.json'),'code_sha256':sha(Path(__file__)),'source_frame_sha256':{f['id']:sha(ROOT/'source_frames'/(f['id']+'.png')) for f in ANN['frames']}}
    (ROOT/'evidence'/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    assert all(c['passed'] for c in checks), checks
    print(json.dumps({'comparisons':len(all_rows),'new_model_calls':0,'summaries':summaries,'checks_passed':len(checks)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
