"""Verify the paired input ablation and score only explicitly user-reviewed targets."""
import base64,csv,hashlib,json,statistics,sys
from pathlib import Path
import numpy as np
from PIL import Image
import run_crop_comparison as vlm
import run_direct_cap as direct
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def main():
    cfg=read(ROOT/'config.json');samples=read(ROOT/'samples.json');gold=read(ROOT/'human_labels.json')['labels']
    v=rows(ROOT/'vlm_raw.jsonl');c=rows(ROOT/'color_raw.jsonl');a=rows(ROOT/'direct_cap/raw.jsonl')
    checks=[]
    def check(name,condition):
        checks.append({'check':name,'passed':bool(condition)})
        if not condition:raise AssertionError(name)
    check('36 VLM paired calls',len(v)==36)
    check('36 classical paired predictions',len(c)==36)
    check('6 separate posthoc calls',len(a)==6)
    check('only six explicit human labels',gold==dict(zip('GHIJKL',[9,1,2,10,5,11])))
    check('identical text across all 36 paired calls',len({json.dumps(r['messages']) for r in v})==1)
    check('fixed seed/settings',all(r['options']==cfg['options'] for r in v+a))
    check('no paired retries',all(r['attempt']==1 for r in v+a))
    check('independent paired calls',all(len(r['messages'])==2 for r in v+a))
    ledger=rows(ROOT/'vlm_request_ledger.jsonl');check('all intents have saved outcomes',[r['request_sha256'] for r in ledger]==[r['request_sha256'] for r in v])
    for s in samples:
        sid=s['id'];imgs={k:np.asarray(Image.open(ROOT/m['file'])) for k,m in s['inputs'].items()}
        check(sid+' identical dimensions',all(im.shape==(224,336,3) for im in imgs.values()))
        check(sid+' same cap pixels H vs HS',np.array_equal(imgs['H'][:,:168],imgs['HS'][:,:168]))
        check(sid+' same silks pixels S vs HS',np.array_equal(imgs['S'][:,168:],imgs['HS'][:,168:]))
        check(sid+' no cap panel in S',np.all(imgs['S'][:,:168]==96))
        check(sid+' no silks panel in H',np.all(imgs['H'][:,168:]==96))
        frame=Image.open(ROOT/s['source_file']).convert('RGB')
        check(sid+' original frame unchanged',sha((ROOT/s['source_file']).read_bytes())==s['source_sha256'])
        check(sid+' cap crop exactly from original',np.array_equal(np.asarray(frame.crop(s['cap_box'])),np.asarray(Image.open(ROOT/s['cap_file']))))
        tor=frame.crop(s['torso_box'])
        for m in s['torso_masked_helmet_regions']:
            b=m['source_box'];tb=s['torso_box'];tor.paste((96,96,96),[b[0]-tb[0],b[1]-tb[1],b[2]-tb[0],b[3]-tb[1]])
        check(sid+' shirt mask reconstructed',np.array_equal(np.asarray(tor),np.asarray(Image.open(ROOT/s['torso_file']))))
        for condition in ['H','S','HS']:
            r=next(r for r in v if r['sample']==sid and r['condition']==condition)
            body,_=vlm.build(s,condition,cfg)
            check(sid+condition+' exact payload hash',sha(json.dumps(body,ensure_ascii=False).encode())==r['request_sha256'])
            check(sid+condition+' input whitelist excludes source overlays and labels',[m['file'] for m in r['images']]==['references/registry.png',s['inputs'][condition]['file']])
    for ref in read(ROOT/'reference_manifest.json'):
        check('reference '+str(ref['number'])+' unchanged',sha((ROOT/ref['file']).read_bytes())==ref['sha256'])
    for r in a:
        data=(ROOT/r['image']).read_bytes();msgs=[{'role':m['role'],'content':m['content']} for m in r['messages']];msgs[1]['images']=[base64.b64encode(data).decode()]
        body={'model':cfg['model'],'messages':msgs,'stream':False,'think':False,'keep_alive':'10m','options':cfg['options'],'format':direct.SCHEMA}
        check(r['sample']+' direct exact request hash',sha(json.dumps(body,ensure_ascii=False).encode())==r['request_sha256'])
        check(r['sample']+' deterministic class-to-number mapping',direct.CLASSES.index(r['parsed']['color_class'])==r['number'])
    allrows=[]
    for method,records in [('VLM',v),('HSV',c),('VLM_direct_cap',a)]:
        for r in records:
            sid=r['sample'];prediction=r.get('number',r.get('parsed',{}).get('number'));confirmed=sid in gold
            allrows.append({'method':method,'condition':r.get('condition','direct_cap'),'sample':sid,'human_gold':gold.get(sid,''),'human_verified':confirmed,'prediction':prediction,'correct':prediction==gold[sid] if confirmed else '', 'wrong_assertion':prediction not in [0,gold[sid]] if confirmed else '', 'abstained':prediction==0,'latency_ms':r['kernel_median_ms'] if method=='HSV' else r['http_elapsed_ms'],'timing_scope':'in-memory CPU feature+score' if method=='HSV' else 'local HTTP response including model work','failure':r.get('failure',False),'raw_output':json.dumps(r.get('parsed',{'top1':r.get('top1_number'),'score':r.get('top_score'),'margin':r.get('margin')}),ensure_ascii=False)})
    with (ROOT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    metrics=[]
    for method,condition in [('VLM','H'),('VLM','S'),('VLM','HS'),('HSV','H'),('HSV','S'),('HSV','HS'),('VLM_direct_cap','direct_cap')]:
        group=[r for r in allrows if r['method']==method and r['condition']==condition and r['human_verified']]
        n=len(group);correct=sum(r['correct'] for r in group);wrong=sum(r['wrong_assertion'] for r in group);abstained=sum(r['abstained'] for r in group)
        metrics.append({'method':method,'condition':condition,'n':n,'correct':correct,'wrong':wrong,'abstained':abstained,'overall_accuracy':correct/n,'coverage':(n-abstained)/n,'accuracy_when_answered':correct/(n-abstained) if n>abstained else None,'median_ms':statistics.median(r['latency_ms'] for r in group),'p95_ms':float(np.percentile([r['latency_ms'] for r in group],95)),'timing_scope':group[0]['timing_scope']})
    with (ROOT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    summary={'human_verified_samples':len(gold),'unverified_samples_excluded_from_accuracy':list('ABCDEF'),'paired_vlm_calls':len(v),'posthoc_vlm_calls':len(a),'total_vlm_calls':len(v)+len(a),'classical_predictions':len(c),'classical_benchmark_repeats_per_prediction':200,'training_steps':0,'jev_calls':0,'vlm_failures':sum(r.get('failure',False) for r in v+a),'paired_schema_violations':sum(not r['format_valid'] for r in v),'paired_H_abstentions_all_12':sum(r['parsed']['number']==0 for r in v if r['condition']=='H'),'S_claims_cap_class_despite_hidden_cap':sum(r['parsed']['cap_class']!='not_visible' for r in v if r['condition']=='S'),'paired_vlm_total_http_seconds':sum(r['http_elapsed_ms'] for r in v)/1000,'posthoc_total_http_seconds':sum(r['http_elapsed_ms'] for r in a)/1000,'first_call_load_seconds':v[0]['response_json']['load_duration']/1e9,'metrics':metrics,'checks_passed':len(checks),'limits':['One race; scored six riders in one frame. Twelve crops from two frames are correlated, not twelve independent race scenarios.','Only manually located ROIs tested. No detector, camera-transition, tracking, leader-ranking or commentary accuracy measured.','All H/S/HS conditions use the same canvas and registry. The posthoc direct-cap stage changes several things together and is not a matched causal comparison.','Legacy HSV scores are not probabilities and thresholds were not tuned on these six labels. No new classifier was trained.','CPU kernel timing and VLM HTTP timing have different boundaries. Neither is complete video-to-commentary latency.','Only numbers 1,2,5,9,10,11 are in the human-verified sample; other classes and cameras remain untested.']}
    write(ROOT/'summary.json',summary);write(ROOT/'verification.json',checks)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
