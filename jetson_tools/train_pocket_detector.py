"""Local pocket detector training; no uploads or implicit model downloads."""
import argparse,os
from pathlib import Path
root=Path(__file__).resolve().parent
os.chdir(root)
p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=20);p.add_argument('--refine',action='store_true');args=p.parse_args()
import torch
from ultralytics import YOLO
torch.set_num_threads(2)
model=YOLO(str(root/('training/pockets/weights/best.pt' if args.refine else 'yolo26n.pt')))
model.train(data=str(root/'datasets/pockets/data.yaml'),epochs=args.epochs,imgsz=512 if args.refine else 416,batch=2 if args.refine else 4,
            device=0,workers=0,amp=False,cache=False,plots=False,project=str(root/'training'),
            name='pockets_refined' if args.refine else 'pockets',exist_ok=True,patience=8,close_mosaic=0,save=True,seed=0)
