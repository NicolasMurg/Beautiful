import math
import numpy as np
from perception_quality import closest_pair, DepthSmoother, depth_quality


def snap(n,t, recovering=None):
    return ('image',n,0,t,None,recovering)


def rec(z=1.,box=(10,10,110,110),label='bottle'):
    return dict(depth_m=z,bbox_xyxy=list(box),object=label)


def test_pair_improves_gap_without_reusing_or_taking_stale_frames():
    histories=[[snap(1,.90),snap(2,.94),snap(3,.98)], [snap(1,.941),snap(2,.999)]]
    pair=closest_pair(histories,[0,0],1.)
    assert [s[1] for s in pair]==[2,1]
    assert abs(pair[0][3]-pair[1][3])<.002
    pair=closest_pair(histories,[2,1],1.)
    assert [s[1] for s in pair]==[3,2]
    assert closest_pair(histories,[3,2],1.) is None
    assert closest_pair(histories,[0,0],2.) is None


def test_pair_rejects_bad_gap_recovery_and_future_timestamps():
    assert closest_pair([[snap(1,.9)],[snap(1,.99)]],[0,0],1.) is None
    assert closest_pair([[snap(1,.99, recovering=.99)],[snap(1,.99)]],[0,0],1.) is None
    assert closest_pair([[snap(1,1.01)],[snap(1,1.01)]],[0,0],1.) is None


def test_smoothing_reduces_small_jitter_and_retains_raw():
    s=DepthSmoother()
    s.update([rec(1.)],1.,1.01)
    r=rec(1.04);s.update([r],1.1,1.11)
    assert r['depth_m']==1.04 and r['depth_smoothed_m']==1.02
    r=rec(.98);s.update([r],1.2,1.21)
    assert r['depth_smoothed_m']==1. and r['smoothing_samples']==3


def test_invalid_missing_and_stale_clear_history():
    for value in [None,float('nan'),float('inf'),0,-1]:
        s=DepthSmoother();s.update([rec()],1.,1.01)
        r=rec(value);s.update([r],1.1,1.11)
        assert r['depth_smoothed_m'] is None
        r=rec(1.04);s.update([r],1.2,1.21)
        assert r['smoothing_samples']==1
    s=DepthSmoother();s.update([rec()],1.,1.01)
    s.update([],1.1,1.11)
    r=rec();s.update([r],1.2,1.21);assert r['smoothing_samples']==1
    r=rec();s.update([r],1.3,1.7)
    assert r['depth_smoothed_m'] is None and r['smoothing_status']=='stale_frame'
    r=rec();s.update([r],1.8,1.81);assert r['smoothing_samples']==1


def test_motion_class_and_depth_changes_start_new_history():
    for r,reason in [(rec(1.,(30,10,130,110)),'object_moved'),(rec(1.3),'depth_changed'),(rec(1.04,label='cup'),'new_object')]:
        s=DepthSmoother();s.update([rec()],1.,1.01)
        s.update([r],1.1,1.11)
        assert r['smoothing_samples']==1 and r['smoothing_status']==reason


def test_ambiguous_same_class_objects_do_not_share_history():
    s=DepthSmoother();s.update([rec()],1.,1.01)
    records=[rec(1.02),rec(1.06,(12,10,112,110))]
    s.update(records,1.1,1.11)
    assert all(r['smoothing_samples']==1 for r in records)
    r=rec();s.update([r],1.2,1.21)
    assert r['smoothing_samples']==1 and r['smoothing_status']=='ambiguous_object'


def test_appearance_change_and_timeout_reset_history():
    s=DepthSmoother();sig=np.zeros((8,8));s.update([rec()],1.,1.01,[sig])
    r=rec(1.04);s.update([r],1.1,1.11,[sig+100])
    assert r['smoothing_status']=='appearance_changed' and r['smoothing_samples']==1
    r=rec();s.update([r],1.6,1.61,[sig+100]);assert r['smoothing_samples']==1


def test_quality_reports_sparse_and_mixed_depth():
    z=np.full((100,100),np.nan)
    z[40:50,40:50]=1
    q=depth_quality(z,[0,0,100,100])
    assert q['valid_depth_count']==100 and q['quality_status']=='insufficient_stereo_depth'
    z[:]=1;z[50:]=2
    q=depth_quality(z,[0,0,100,100])
    assert q['quality_status']=='mixed_depth_region' and q['valid_depth_fraction']==1
    z[:]=1
    assert depth_quality(z,[0,0,100,100])['quality_status']=='valid_depth'


def test_camera_history_bounded_and_cleared_after_dropout():
    import threading
    from dual_camera_preview import Camera
    class Capture:
        def __init__(self,owner):self.owner=owner;self.index=0
        def read(self):
            self.index+=1
            if self.index==5:return False,None
            if self.index==9:self.owner.stop.set()
            return True,np.zeros((2,2,3),np.uint8)
        def release(self):pass
    class TestCamera(Camera):
        def open_capture(self):
            if not hasattr(self,'fake'):self.fake=Capture(self)
            return self.fake
    stop=threading.Event();c=TestCamera(None,'Jieli','fake','MJPG',2,2,25,stop,history_size=2)
    c.capture()
    history=c.history_snapshot()
    assert len(history)==2 and all(h[1]>=5 for h in history)
    assert c.error is None
