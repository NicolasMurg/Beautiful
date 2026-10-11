"""Evaluate local checkpoints on held-out pocket images without uploading data."""
from pathlib import Path
import argparse,json,torch
from ultralytics import YOLO
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--refined',action='store_true')
a=p.parse_args()
torch.set_num_threads(2)
name='pockets_refined' if a.refined else 'pockets'
model=YOLO(str(root/'training'/name/'weights/best.pt'))
r=model.val(data=str(root/'datasets/pockets/data.yaml'),split='test',imgsz=512 if a.refined else 416,batch=2,device=0,workers=0,plots=False,project=str(root/'training'),name=name+'-test',exist_ok=True)
(root/('pocket-refined-test-metrics.json' if a.refined else 'pocket-test-metrics.json')).write_text(json.dumps(r.results_dict,indent=2))
