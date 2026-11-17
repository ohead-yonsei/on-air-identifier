"""Evidence-linked assistant review; not expert gold, not an automatic truth judge."""
import csv,hashlib,json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
rows=[json.loads(x) for x in (ROOT/'raw_responses.jsonl').read_text().splitlines()]
A={}
def claim(text,status,evidence,kind='dynamic'):
 return {'claim':text,'status':status,'kind':kind,'evidence':evidence}
def put(s,c,claims,note='',internal=False,irrelevant=0):A[s,c]={'claims':claims,'note':note,'internal_order_body_conflict':internal,'off_context_or_generic_flag':irrelevant}
U='unverified';T='supported_live';F='contradicted_dynamic';BG='verified_background'
def ranks(ns,clear=False):
 return [claim(f'No.{n} is rank {i+1}', T if clear and i==0 and n==9 else F if clear and i==0 and n!=9 else U,
   'Masked sequence shows a sustained purple-cap No.9 lead.' if clear and i==0 else 'No independent timestamp-aligned rank label; do not substitute checkpoint ranks.') for i,n in enumerate(ns)]
def close(ns):return [claim(f'No.{n} is close behind the stated leader',U,'Exact proximity/adjacency not certified in the limited visual reference.') for n in ns]
for c in ['F','B']:
 put('S1_early',c,[claim('The field is leaving the start',T,'S1 input frames 1–16 show the gates and the opening run.'),claim('The pack is tightly bunched',U,'Tightness is not operationally defined; two panels and overlap complicate depth.'),claim('No single leader has emerged',U,'ORDER unclear is abstention; the separate assertion that no leader exists is not established.')],'Correct start context; uncertainty did not lead to background facts.')
put('S1_early','F-shuffle',ranks([1,11,3])+close([11,3])+[claim('The field is on the straight',T,'S1 opening sequence shows a straight departure from the stalls.')],'No expert leader/top3 label; asserted numbers remain unverified.')
for s in ['S1_early','S2_bend','S3_wide','S4_straight']:
 put(s,'F-no-video',[claim('The finish is too close to call',U,'No images supplied; a future finish outcome cannot be established from the card.'),claim('The leaders are jostling for position',U,'No dynamic evidence supplied.'),claim('There is no clear gap between the top contenders',F if s in ['S2_bend','S4_straight'] else U,'S2/S4 masked frames show an isolated purple-cap leader.' if s in ['S2_bend','S4_straight'] else 'No dynamic evidence supplied; ambiguous scene has no certified gap label.'),claim('The field is sprinting toward the line now',U,'Generic finish narrative; card and total race distance do not establish current location.')],'Identical answer in all four no-video calls; ORDER unclear coexists with unsupported live claims.',irrelevant=1)
for c,ns in [('F',[1,6,7]),('F-shuffle',[1,11,6])]:
 put('S2_bend',c,ranks(ns,True)+close(ns[1:])+[claim('A finish is happening now',U,'The sequence is a bend; the phrase thrilling finish is not evidence of current finish status.')],'No.9 visibly leads. Rank2/3 are left unverified instead of inventing full gold.',irrelevant=1)
put('S2_bend','B',ranks([6,1,7],True)+close([1,7])+[claim('The field rounds a bend',T,'Curved rail and turning trajectories in S2 frames.'),claim('This is the final bend',U,'Final-bend designation not independently time-aligned; no checkpoint rank is used.')],'Wrong leader persists; no background transition.')
put('S3_wide','F',ranks([10,11,5])+close([5])+[claim('The field rounds a bend',T,'Visible curved rail and turn-to-side camera sequence.')],'Leader/top3 not certified at final distant frames. The body phrase lead the pack is treated as a lead-group paraphrase, not an extra duplicate claim.')
put('S3_wide','B',ranks([10,11,5])+close([5])+[claim('The field rounds a bend',T,'Visible curved rail and turn-to-side camera sequence.'),claim('The field heads down the straight now',U,'Latest side view alone does not certify the exact course section.')],'B keeps the same ORDER as F and adds a location assertion; no background transition.')
put('S3_wide','F-shuffle',ranks([9,10,2])+close([10,2])+[claim('The field rounds a bend',T,'Visible curved rail and turn-to-side camera sequence.'),claim('No.9 has a purple cap',BG,'Supplied racecard No.9.',kind='background'),claim('No.10 has a sky blue cap',BG,'Supplied racecard No.10.',kind='background'),claim('No.2 has a yellow cap',BG,'Supplied racecard No.2.',kind='background')],'Three verified cap associations are inline identification descriptions, not a fallback passage. Correct leader possibility is not scored as gold for S3.')
for c,ns in [('F',[1,2,6]),('B',[10,2,6])]:
 put('S4_straight',c,ranks(ns,True)+close(ns[1:])+[claim('The field is rounding a bend now',U,'The interval contains a bend-to-straight transition; exact current section not certified.')],'Purple-cap No.9 visibly leads; no verified background facts.')
put('S4_straight','F-shuffle',ranks([9,6,2],True)+[claim('No.6 Purple Sinhwa leads (body)',F,'Body contradicts visible purple-cap No.9 and its own ORDER first number9.')]+close([5,2])+[claim('The field is rounding a bend now',U,'Bend-to-straight transition; exact current section not certified.')],'ORDER9 is supported, but prose names6 as leader. The two conflicting claims are retained separately.',internal=True)
assert len(A)==16 and len(rows)==16
scored=[];claimrows=[]
for r in sorted(rows,key=lambda r:(r['scene'],['F','F-shuffle','F-no-video','B'].index(r['condition']))):
 ann=A[r['scene'],r['condition']]; cs=ann['claims'];order=r['format']['order']
 z={'scene':r['scene'],'condition':r['condition'],'ORDER':'unclear' if order==[] else ','.join(map(str,order or [])),'confirmed_wrong_dynamic_claims':sum(x['status']==F for x in cs),'unsupported_assertions':sum(x['status']==U for x in cs),'supported_live_claims':sum(x['status']==T for x in cs),'verified_background_facts':sum(x['status']==BG for x in cs),'within_response_repetition':0,'cross_response_identical':r['condition']=='F-no-video','off_context_or_generic_flag':ann['off_context_or_generic_flag'],'internal_order_body_conflict':ann['internal_order_body_conflict'],'elapsed_seconds':r['elapsed_seconds'],'load_seconds':r['response_json'].get('load_duration',0)/1e9,'failure':r['failure'],'format_issues':';'.join(r['format']['issues']),'image_count':r['image_count'],'prompt_eval_count':r['response_json']['prompt_eval_count'],'response':r['text'],'review_note':ann['note'],'rating_status':'assistant_visual_review_requires_human_validation'}
 scored.append(z)
 for i,c in enumerate(cs):claimrows.append({'scene':r['scene'],'condition':r['condition'],'claim_index':i+1,**c,'human_review':'','evidence_file':f"frames/{r['scene']}/contact.jpg"})
with (ROOT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(scored[0]));w.writeheader();w.writerows(scored)
with (ROOT/'claim_review.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(claimrows[0]));w.writeheader();w.writerows(claimrows)
(ROOT/'scored_responses.jsonl').write_text(''.join(json.dumps({'scene':r['scene'],'condition':r['condition'],**A[r['scene'],r['condition']]},ensure_ascii=False)+'\n' for r in rows))
summary={}
for c in ['F','F-shuffle','F-no-video','B']:
 rr=[r for r in scored if r['condition']==c]
 summary[c]={'calls':len(rr),'ORDER_unclear':sum(r['ORDER']=='unclear' for r in rr),'median_seconds':statistics.median(r['elapsed_seconds'] for r in rr),'total_seconds':sum(r['elapsed_seconds'] for r in rr),**{k:sum(r[k] for r in rr) for k in ['confirmed_wrong_dynamic_claims','unsupported_assertions','supported_live_claims','verified_background_facts','failure']}}
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
