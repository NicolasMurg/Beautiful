from pool_score import AutoScore

P=[dict(id=0,center=[80,50],radius=10)]

def approach(score,group='solids'):
    for n,x in enumerate((20,30,40,55,70)):
        score.update([dict(center=[x,50],radius=5,group=group)],P,n*.3)


def lose(score,pockets=P,clear=True):
    return [score.update([],pockets,t,clear) for t in (2,2.3,2.8)]


def test_automatic_point_once_persistent_and_undo(tmp_path):
    s=AutoScore(tmp_path/'score.json');approach(s);lose(s)
    assert s.totals['solids']==1 and s.totals['stripes']==0
    s.update([],P,3.2);assert s.totals['solids']==1
    restored=AutoScore(s.path);assert restored.totals==s.totals
    restored.undo();assert restored.totals['solids']==0
    restored.reset();assert not restored.events


def test_unknown_occlusion_or_missing_pocket_do_not_score(tmp_path):
    for group,clear,pockets in [(None,True,P),('stripes',False,P),('solids',True,[])]:
        s=AutoScore(tmp_path/f'{group}-{clear}.json');approach(s,group);lose(s,pockets,clear)
        assert sum(s.totals.values())==0


def test_disappearance_away_from_pocket_not_a_pot(tmp_path):
    s=AutoScore(tmp_path/'score.json')
    for n in range(5):s.update([dict(center=[20,50],radius=5,group='stripes')],P,n*.3)
    lose(s);assert sum(s.totals.values())==0


def test_scratch_and_eight_never_award_group_point(tmp_path):
    for group in ('scratch','eight'):
        s=AutoScore(tmp_path/f'{group}.json');approach(s,group);lose(s)
        assert s.totals[group]==1 and s.totals['solids']==s.totals['stripes']==0


def test_color_group_estimates_and_small_unknown():
    import numpy as np,cv2
    from pool_score import appearance_group
    frame=np.full((60,60,3),(45,90,30),np.uint8)
    box=[15,15,45,45]
    cv2.circle(frame,(30,30),15,(0,0,255),-1)
    assert appearance_group(frame,box)=='solids'
    cv2.rectangle(frame,(15,25),(45,35),(255,255,255),-1)
    assert appearance_group(frame,box)=='stripes'
    cv2.circle(frame,(30,30),15,(250,250,250),-1)
    assert appearance_group(frame,box)=='scratch'
    cv2.circle(frame,(30,30),15,(10,10,10),-1)
    assert appearance_group(frame,box)=='eight'
    assert appearance_group(frame,[26,26,34,34]) is None


def test_pocket_locations_require_repeated_agreement():
    from pocket_locations import StablePockets
    s=StablePockets();p=[dict(id=0,center=[50,50],radius=8,confidence=.5)]
    assert s.update(p)==[] and s.update(p)==[]
    assert len(s.update(p))==1
    assert s.update([])==[]


def test_stale_gap_and_reappeared_ball_suppress_pot(tmp_path):
    s=AutoScore(tmp_path/'gap.json');approach(s)
    s.update([],P,4.0);s.update([],P,4.3);s.update([],P,4.6)
    assert sum(s.totals.values())==0
    s=AutoScore(tmp_path/'reappeared.json');approach(s)
    s.update([],P,2.0);s.update([],P,2.3)
    s.update([dict(center=[70,50],radius=5,group='solids')],P,2.8)
    assert sum(s.totals.values())==0
