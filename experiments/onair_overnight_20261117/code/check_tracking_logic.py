"""Small checks for missing candidates, exclusivity, reset, and causal warp."""
import json
from common import *
from tracking import associate,Tracker
def d(i,c,b):return {'id':i,'class':c,'box':b,'score':.9}
checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
h=d('h0','horse',[0,50,100,150]);near=d('p0','person',[30,20,70,80]);far=d('p1','person',[400,400,450,500])
p,u=associate([h,far]);check('no_candidate_stays_unassigned',p[0]['person'] is None and u==['p1'])
p,u=associate([h,near,d('h1','horse',[10,50,110,150])]);check('one_person_not_assigned_twice',sum(x['person'] is not None for x in p)==1)
t=Tracker(True);x=t.step([h],{},True);y=t.step([h],{},False);check('stationary_id_is_preserved',x['detection_to_track']==y['detection_to_track'])
z=t.step([h],{},True);check('cut_resets_identity',x['detection_to_track']!=z['detection_to_track'])
t=Tracker(True);t.step([h],{},True);moved=d('h0','horse',[300,50,400,150]);y=t.step([moved],{'affine':[[1,0,300],[0,1,0]]},False);check('background_translation_applied_before_match',y['detection_to_track']['h0']==1)
for _ in range(4):t.step([],{},False)
check('missing_track_expires_after_three',len(t.active)==0)
dump(ROOT/'evidence/tracking_logic_checks.json',{'time_utc':now(),'checks':checks,'scope':'synthetic implementation invariants, not racing accuracy'})
print(json.dumps({'checks_passed':len(checks)}))
