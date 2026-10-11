"""Conservative automatic pot estimates from stable groups and pocket approaches."""
from collections import Counter,deque
import json,math,uuid
from pathlib import Path


def appearance_group(frame,box):
    """Conservative white-band/color fractions, not ball-number recognition."""
    import cv2
    import numpy as np
    x1,y1,x2,y2=box;cx,cy=(x1+x2)/2,(y1+y2)/2
    r=min(x2-x1,y2-y1)/2
    if r<5:return None
    h,w=frame.shape[:2]
    xa,xb=max(0,int(cx-r)),min(w,math.ceil(cx+r))
    ya,yb=max(0,int(cy-r)),min(h,math.ceil(cy+r))
    yy,xx=np.mgrid[ya:yb,xa:xb]
    mask=(xx+.5-cx)**2+(yy+.5-cy)**2<=(r*.86)**2
    hsv=cv2.cvtColor(frame[ya:yb,xa:xb],cv2.COLOR_BGR2HSV)[mask]
    if len(hsv)<30:return None
    _,sat,val=hsv.T
    white=float(np.mean((sat<55)&(val>140)))
    color=float(np.mean((sat>85)&(val>65)))
    black=float(np.mean(val<55))
    if white>.88 and color<.05 and black<.05:return 'scratch'
    if black>.80 and color<.05 and white<.15:return 'eight'
    if color>.65 and white<.20:return 'solids'
    if .32<white<.80 and color>.22:return 'stripes'
    return None


class AutoScore:
    def __init__(self,path):
        self.path=Path(path);self.tracks={};self.next_id=0;self.events=[]
        self.totals={'solids':0,'stripes':0,'scratch':0,'eight':0}
        self.session=uuid.uuid4().hex[:8];self.last_time=None
        try:
            state=json.loads(self.path.read_text())
            for k in self.totals:self.totals[k]=int(state['totals'][k])
            self.events=state.get('events',[])
        except (OSError,ValueError,KeyError,TypeError):pass

    def save(self):
        temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(totals=self.totals,events=self.events,mode='automatic_estimates'),indent=2))
        temp.replace(self.path)

    def reset(self):
        self.tracks.clear();self.events=[]
        self.totals={k:0 for k in self.totals};self.save()

    def undo(self):
        if self.events:
            event=self.events.pop();self.totals[event['group']]-=1
            self.tracks.clear();self.save()

    def update(self,observations,pockets,now,scene_clear=True):
        """Each observation has center/radius/group. Only first camera scores.

        Occlusion, unknown groups, competing tracks or gaps suppress credit.
        Pot estimate requires stable label, approach into a pocket, disappearance.
        """
        if self.last_time is not None and (now<=self.last_time or now-self.last_time>2):self.tracks.clear()
        self.last_time=now
        if not scene_clear:
            self.tracks.clear();return []
        eligible=[(key,t) for key,t in self.tracks.items() if now-t['seen']<1.5]
        matches={}
        for i,o in enumerate(observations):
            candidates=[key for key,t in eligible if math.dist(t['center'],o['center'])<max(35,o['radius']*5)
                        and .55<t['radius']/max(1,o['radius'])<1.8]
            if len(candidates)==1:matches[i]=candidates[0]
        for i,key in list(matches.items()):
            if list(matches.values()).count(key)>1:matches.pop(i,None)
        touched=set()
        for i,o in enumerate(observations):
            key=matches.get(i)
            if key is None:
                self.next_id+=1;key=f'{self.session}-{self.next_id}'
                self.tracks[key]=dict(center=o['center'],radius=o['radius'],seen=now,groups=deque(maxlen=8),
                                      history=deque(maxlen=8),armed=None,misses=0)
            t=self.tracks[key];t['center']=o['center'];t['radius']=o['radius'];t['seen']=now;t['misses']=0
            t['groups'].append(o.get('group'));t['history'].append(o['center']);touched.add(key)
            votes=Counter(t['groups']);group,n=votes.most_common(1)[0]
            t['group']=group if group is not None and len(t['groups'])>=4 and n/len(t['groups'])>=.8 else None
            t['armed']=None
            for p in pockets:
                d=math.dist(o['center'],p['center']);limit=p['radius']+o['radius']*.6
                previous=list(t['history'])[:-1]
                if d<limit and previous and max(math.dist(v,p['center']) for v in previous)>limit+o['radius']*.5:
                    t['armed']=dict(p);break
        emitted=[]
        for key,t in list(self.tracks.items()):
            if key in touched:continue
            t['misses']+=1
            if now-t['seen']>1.5 and t['misses']>=3:
                group=t.get('group')
                if t['armed'] is not None and any(math.dist(p['center'],t['armed']['center'])<max(p['radius'],t['armed']['radius']) for p in pockets) and group is not None and not any(math.dist(o['center'],t['center'])<max(35,t['radius']*5) for o in observations) and (group not in ('solids','stripes') or self.totals[group]<7) and (group!='eight' or self.totals[group]<1):
                    event=dict(track=key,group=group,pocket=t['armed']['id'],pocket_center=t['armed']['center'],evidence='stable_appearance_pocket_approach_then_loss')
                    self.totals[group]+=1;self.events.append(event);emitted.append(event);self.save()
                del self.tracks[key]
        return emitted
