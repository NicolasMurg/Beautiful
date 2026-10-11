from types import SimpleNamespace
import numpy as np
from ball_perception import ball_depth, detail_box_records

G=SimpleNamespace(fx=100,fy=100)


def test_small_fully_matched_ball_no_longer_impossible():
    r=ball_depth(np.full((60,80),2.,np.float32),[20,20,27,27],G)
    assert 12<=r['depth_sample_count']<30
    assert r['valid_depth_fraction']==1
    assert r['depth_m']==2
    assert r['projected_bbox_width_m']==.14


def test_unknown_for_missing_too_small_and_mixed_depth():
    for depth,box,status in [(np.full((60,80),np.nan),[20,20,28,28],'insufficient_stereo_depth'),
                             (np.ones((60,80)),[20,20,23,23],'ball_too_small')]:
        r=ball_depth(depth,box,G);assert r['depth_m'] is None and r['quality_status']==status
    depth=np.ones((60,80));depth[:,24:]=3
    assert ball_depth(depth,[20,20,28,28],G)['quality_status']=='mixed_depth_region'


def test_single_match_never_becomes_a_distance():
    depth=np.full((60,80),np.nan);depth[23,23]=2
    r=ball_depth(depth,[20,20,28,28],G)
    assert r['valid_depth_count']==1 and r['depth_m'] is None


def test_detail_offsets_filters_duplicates_and_scores():
    import torch
    from ultralytics.engine.results import Results
    boxes=torch.tensor([[2,2,10,10,.6,0],[15,5,23,13,.7,0],[24,4,30,12,.9,1]])
    result=Results(np.zeros((40,40,3),np.uint8),path='test',names={0:'billiard ball',1:'bottle'},boxes=boxes)
    old=[dict(object='billiard ball',bbox_xyxy=[12,22,20,30])]
    rows=detail_box_records(result,(10,20),old,{0},np.ones((80,80)),7,.2)
    assert len(rows)==1 and rows[0]['bbox_xyxy']==[25,25,33,33]
    assert abs(rows[0]['detection_confidence']-.7)<1e-6
    assert rows[0]['detection_method']=='yolo_detail'
    assert detail_box_records(result,(10,20),old,set(),np.ones((80,80)),7,.2)==[]


def test_detail_region_clips_edges_and_focuses_small_balls():
    from ball_perception import detail_region
    from ultralytics.engine.results import Results
    import torch
    frame=np.zeros((480,640,3),np.uint8)
    r=Results(frame,path='test',names={0:'billiard ball'},boxes=torch.tensor([[2,2,10,10,.5,0]]))
    assert detail_region(frame.shape,r)==(0,0,160,160)
    r=Results(frame,path='test',names={0:'billiard ball'},boxes=torch.tensor([[300,280,320,300,.5,0]]))
    assert detail_region(frame.shape,r)==(230,210,390,370)
    r=Results(frame,path='test',names={0:'billiard ball'},boxes=torch.empty((0,6)))
    assert detail_region(frame.shape,r)==(192,216,448,408)
