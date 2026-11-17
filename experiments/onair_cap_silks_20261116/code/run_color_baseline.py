"""Frozen legacy appearance scores on manual ROIs; no fitting or detector timing."""
import os
os.environ['FX']='brown,rel47,edge,unknown,small'
import json,time,platform
from pathlib import Path
import cv2,numpy as np
import legacy_color as old
ROOT=Path(__file__).resolve().parents[1]

def features(cap,torso):
    # The same fixed foreground rule and colour table in every condition.
    ch,ck=old.hist(old.quant(cap),old.fgmask(cap).astype(float)+1e-9)
    th,tk=old.hist(old.quant(torso),old.fgmask(torso).astype(float)+1e-9,torso=True)
    return ch,ck,th,tk

def classify(cap,torso,refs,condition):
    # Each ablation only computes/reads features it is allowed to use.
    cs=np.zeros(11);ss=np.zeros(11);ck=tk=0.0
    if condition in ['H','HS']:
        ch,ck=old.hist(old.quant(cap),old.fgmask(cap).astype(float)+1e-9)
        cs=np.array([old.cap_score(ch,n,None)*min(1.,ck*4) for n in range(1,12)])
    if condition in ['S','HS']:
        th,tk=old.hist(old.quant(torso),old.fgmask(torso).astype(float)+1e-9,torso=True)
        ss=np.array([(old.soft(th,r)+old.soft(r,th))/2*min(1.,tk*3) for r in refs])
    scores=cs if condition=='H' else ss if condition=='S' else .6*cs+.4*ss
    ranking=np.argsort(-scores,kind='stable');top=int(ranking[0]);margin=float(scores[top]-scores[ranking[1]])
    accepted=bool(scores[top]>=.40 and margin>=.08)
    return {'top1_number':top+1,'number':top+1 if accepted else 0,'top_score':float(scores[top]),'margin':margin,'scores':scores.tolist(),'cap_known_fraction':ck,'silks_known_fraction':tk}

def main():
    cv2.setNumThreads(1)
    samples=json.loads((ROOT/'samples.json').read_text());manifest=json.loads((ROOT/'reference_manifest.json').read_text())
    refs=[old.silk_ref(str(ROOT/r['file'])) for r in manifest]
    output=[]
    for s in samples:
        t=time.perf_counter_ns();cap=cv2.imread(str(ROOT/s['cap_file']));torso=cv2.imread(str(ROOT/s['torso_file']));io=(time.perf_counter_ns()-t)/1e6
        for condition in ['H','S','HS']:
            result=classify(cap,torso,refs,condition)
            timings=[]
            for _ in range(200):
                t=time.perf_counter_ns();classify(cap,torso,refs,condition);timings.append((time.perf_counter_ns()-t)/1e6)
            output.append({'sample':s['id'],'condition':condition,**result,'input_file_load_ms':io,'kernel_median_ms':float(np.median(timings)),'kernel_p95_ms':float(np.percentile(timings,95)),'timing_repetitions':len(timings),'kernel_timings_ms':timings})
    (ROOT/'color_raw.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    info={'method':'legacy HSV named-colour histogram and soft reference match','trained':False,'test_set_tuning':False,'FX':sorted(old.FX),'threshold':.40,'margin_threshold':.08,'fusion_weights':{'cap':.6,'silks':.4},'one_to_one_assignment':False,'detector_used':False,'threads':cv2.getNumThreads(),'machine':platform.machine(),'latency_scope':'in-memory crop colour features+classification; excludes detection, decoding, crop creation, file IO, reference loading, and RPC','timing_repetitions_per_case':200,'note':'Score is not a calibrated probability. The legacy stripe handling uses colour mixtures, not a trained geometric stripe recognizer.'}
    (ROOT/'color_method.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'predictions':len(output),'training_steps':0,'conditions':['H','S','HS']}))

if __name__=='__main__':main()
