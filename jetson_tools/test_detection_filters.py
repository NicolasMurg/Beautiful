import numpy as np
from detection_filters import DetectionFilters, FilterPanel, filter_result


def test_selection_persistence_and_model_names(tmp_path):
    path=tmp_path/'filters.json'
    f=DetectionFilters({0:'bottle',1:'beverage can'},path)
    assert f.snapshot()==(frozenset({0,1}),0)
    f.set_enabled({1})
    assert f.snapshot()==(frozenset({1}),1)
    restored=DetectionFilters({3:'bottle',7:'beverage can',8:'person'},path)
    assert restored.snapshot()[0]=={7,8}
    f.set_enabled(set())
    assert DetectionFilters(f.names,path).snapshot()[0]==set()
    f.set_enabled(f.names)
    assert f.snapshot()[0]=={0,1}


def test_filter_keeps_boxes_masks_aligned():
    from ultralytics.engine.results import Results
    boxes=np.array([[1,1,5,5,.9,0],[2,2,7,7,.8,1]],dtype=np.float32)
    masks=np.stack([np.zeros((8,8),np.float32),np.ones((8,8),np.float32)])
    r=Results(np.zeros((8,8,3),np.uint8),path='test',names={0:'bottle',1:'beverage can'},boxes=boxes,masks=masks)
    filtered=filter_result(r,{1})
    assert filtered.boxes.cls.tolist()==[1]
    assert len(filtered.masks)==1 and float(filtered.masks.data.sum())==64
    assert len(filter_result(r,set()).boxes)==0
    assert len(filter_result(r,{0,1}).boxes)==2


def test_checkbox_search_all_buttons(tmp_path):
    filters=DetectionFilters({0:'bottle',1:'beverage can',2:'cup'},tmp_path/'filters.json')
    changes=[]
    p=FilterPanel(filters,lambda:changes.append(True),lambda:None)
    try:
        p.poll()
        p.variables[0].set(False);p.changed()
        assert filters.snapshot()[0]=={1,2}
        p.search.set('can');p.poll()
        assert [name for name,row in p.rows if row.winfo_manager()]==['beverage can']
        assert filters.snapshot()[0]=={1,2}
        p.select_all(False);assert not filters.snapshot()[0]
        p.select_all(True);assert filters.snapshot()[0]=={0,1,2}
        assert len(changes)==3
    finally:p.close()


def test_ball_threshold_preserves_other_threshold_and_masks():
    from ultralytics.engine.results import Results
    from detection_filters import filter_scores
    boxes=np.array([[1,1,5,5,.21,0],[2,2,7,7,.21,1],[1,1,6,6,.19,0],[1,1,6,6,.30,1]],dtype=np.float32)
    masks=np.stack([np.full((8,8),n,np.float32) for n in range(4)])
    r=Results(np.zeros((8,8,3),np.uint8),path='test',names={0:'billiard ball',1:'bottle'},boxes=boxes,masks=masks)
    out=filter_scores(r,.25,.20)
    assert out.boxes.cls.tolist()==[0,1]
    assert np.allclose(out.boxes.conf,[.21,.30])
    assert out.masks.data[:,0,0].tolist()==[0,3]
    assert len(filter_scores(r,.9,.9).boxes)==0
