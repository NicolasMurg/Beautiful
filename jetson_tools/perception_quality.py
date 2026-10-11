"""Bounded host-time pairing and conservative display smoothing, independent of models."""
import math
import statistics


def closest_pair(histories, previous, now, max_gap_s=.03, max_age_s=.12):
    """Use each frame at most once; never choose stale or recovery frames.

    Host arrival times do not establish synchronized camera exposures.
    """
    fresh = [[s for s in history if s[0] is not None and s[1] > used
              and s[3] is not None and 0 <= now-s[3] <= max_age_s
              and s[4] is None and s[5] is None]
             for history, used in zip(histories, previous)]
    if len(fresh) != 2 or any(not h for h in fresh):
        return None
    candidates = [(abs(a[3]-b[3]), -min(a[3],b[3]), a, b)
                  for a in fresh[0] for b in fresh[1]
                  if abs(a[3]-b[3]) <= max_gap_s]
    if not candidates:
        return None
    _, _, a, b = min(candidates, key=lambda item: item[:2])
    return [a, b]


def depth_quality(depth, box, inner=.4, min_valid=30):
    import numpy as np
    h,w = depth.shape
    x1,y1,x2,y2 = map(float,box)
    cx,cy=(x1+x2)/2,(y1+y2)/2
    xa,xb=max(0,round(cx-(x2-x1)*inner/2)),min(w,round(cx+(x2-x1)*inner/2))
    ya,yb=max(0,round(cy-(y2-y1)*inner/2)),min(h,round(cy+(y2-y1)*inner/2))
    patch=depth[ya:yb,xa:xb] if xb>xa and yb>ya else np.array([])
    vals=patch[np.isfinite(patch)&(patch>0)]
    count=int(vals.size)
    fraction=count/max(1,patch.size)
    spread=float(np.subtract(*np.percentile(vals,[75,25]))) if count else None
    median=float(np.median(vals)) if count else None
    status=('insufficient_stereo_depth' if count<min_valid or fraction<.25 else
            'mixed_depth_region' if spread/median>.15 else 'valid_depth')
    return dict(valid_depth_count=count, valid_depth_fraction=float(fraction),
                depth_iqr_m=spread, depth_relative_iqr=spread/median if median else None,
                quality_status=status)


def box_iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]))
    y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=x*y
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
    return intersection/union if union>0 else 0


def box_moved(a,b):
    aw,ah=max(1,a[2]-a[0]),max(1,a[3]-a[1])
    bw,bh=max(1,b[2]-b[0]),max(1,b[3]-b[1])
    return (abs((a[0]+a[2]-b[0]-b[2])/2)/aw>.1 or
            abs((a[1]+a[3]-b[1]-b[3])/2)/ah>.1 or
            abs(bw/aw-1)>.1 or abs(bh/ah-1)>.1)


def appearance_signature(frame, box, cv2):
    import numpy as np
    h,w=frame.shape[:2]
    x1,y1,x2,y2=box
    patch=frame[max(0,int(y1)):min(h,math.ceil(y2)),max(0,int(x1)):min(w,math.ceil(x2))]
    if not patch.size: return None
    return cv2.resize(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY),(8,8)).astype(np.float32)


class DepthSmoother:
    """Short median for uniquely associated stationary boxes, never an invalid fallback.

    Association is a display heuristic, not a verified object identity tracker.
    Raw depth is retained; only depth_smoothed_m is filtered.
    """
    def __init__(self, window=3, max_age_s=.3, jump_fraction=.1):
        self.window, self.max_age_s, self.jump_fraction=window,max_age_s,jump_fraction
        self.previous=[]

    def clear(self):
        self.previous=[]

    def update(self, records, stamp, now, signatures=None):
        import numpy as np
        signatures=signatures if signatures is not None else [None]*len(records)
        next_states=[]
        for index,(rec,signature) in enumerate(zip(records,signatures)):
            value=rec['depth_m']
            rec.update(depth_smoothed_m=None,smoothing_samples=0,smoothing_status='invalid_depth')
            if not 0<=now-stamp<=self.max_age_s:
                rec['smoothing_status']='stale_frame'
                continue
            if value is None or not math.isfinite(value) or value<=0:
                continue
            candidates=[p for p in self.previous if p['object']==rec['object'] and
                        box_iou(p['box'],rec['bbox_xyxy'])>=.6 and 0<stamp-p['stamp']<=self.max_age_s]
            state=None
            reason='new_object'
            if len(candidates)==1:
                p=candidates[0]
                other_matches=sum(r['object']==p['object'] and box_iou(r['bbox_xyxy'],p['box'])>=.6 for r in records)
                if other_matches!=1:
                    reason='ambiguous_object'
                elif box_moved(p['box'],rec['bbox_xyxy']):
                    reason='object_moved'
                elif signature is not None and p['signature'] is not None and np.mean(abs(signature-p['signature']))>20:
                    reason='appearance_changed'
                elif abs(value-p['values'][-1][1])/p['values'][-1][1]>self.jump_fraction:
                    reason='depth_changed'
                else:
                    state=p
            elif len(candidates)>1:
                reason='ambiguous_object'
            values=[v for v in state['values'] if stamp-v[0]<=self.max_age_s] if state else []
            values=(values+[(stamp,float(value))])[-self.window:]
            rec.update(depth_smoothed_m=statistics.median(v for _,v in values),
                       smoothing_samples=len(values), smoothing_status='smoothed' if len(values)>1 else reason)
            next_states.append(dict(object=rec['object'],box=list(rec['bbox_xyxy']),
                                    stamp=stamp,values=values,signature=signature))
        self.previous=next_states
        return records
