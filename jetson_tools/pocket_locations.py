"""Pocket proposals with close crops for small openings in a wide table view."""
import math


def detect_pockets(model,frame,device='0',half=True):
    h,w=frame.shape[:2]
    y=int(h*.4);width=w//2
    offsets=[(0,0),(0,y),(w//4,y),(w-width,y)]
    crops=[frame]+[frame[y:h,x:x+width] for x,_ in offsets[1:]]
    results=model.predict(crops,device=device,imgsz=640,conf=.20,verbose=False,
                          quantize=16 if device!='cpu' and half else None)
    proposals=[]
    for result,(ox,oy) in zip(results,offsets):
        for b,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
            x1,y1,x2,y2=b
            center=[(x1+x2)/2+ox,(y1+y2)/2+oy]
            radius=max(6,math.sqrt((x2-x1)*(y2-y1))/2)
            if radius>min(h,w)*.15:continue
            proposals.append(dict(center=center,radius=radius,confidence=float(score)))
    selected=[]
    for p in sorted(proposals,key=lambda p:p['confidence'],reverse=True):
        if any(math.dist(p['center'],q['center'])<max(p['radius'],q['radius']) for q in selected):continue
        selected.append(p)
    selected=sorted(selected[:6],key=lambda p:(p['center'][1],p['center'][0]))
    for i,p in enumerate(selected):p['id']=i
    return selected


class StablePockets:
    def __init__(self):self.tracks=[];self.next_id=0
    def update(self,proposals):
        tracks=[];used=set()
        for p in proposals:
            old=next((t for t in self.tracks if t['id'] not in used and math.dist(t['center'],p['center'])<max(12,p['radius'])),None)
            if old is None:
                self.next_id+=1;count=1;key=self.next_id
            else:count=old['count']+1;key=old['id'];used.add(key)
            tracks.append(dict(p,count=count,id=key))
        self.tracks=tracks
        return [dict(t) for t in tracks if t['count']>=3]
