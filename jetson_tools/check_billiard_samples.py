import os,json,time
from pathlib import Path
import cv2,torch
from ultralytics import YOLO
root=Path(__file__).resolve().parent
os.chdir(root);torch.set_num_threads(2)
model=YOLO(str(root/'yoloe-26s-objects.pt'))
frames=[cv2.imread('/tmp/billiard-'+n+'.png') for n in ('left','right')]
results=model.predict(frames,device=0,imgsz=640,conf=0.08,verbose=False,quantize=16)
for name,r in zip(('left','right'),results):
 rows=[dict(label=r.names[int(c)],score=float(s),box=b) for b,c,s in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.cls.cpu().tolist(),r.boxes.conf.cpu().tolist())]
 print(name,json.dumps(rows),flush=True)
 cv2.imwrite('/tmp/billiard-yolo-'+name+'.png',r.plot(masks=False))
