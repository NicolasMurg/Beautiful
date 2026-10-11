"""Prepare cached, text-prompted YOLOE weights; no can training or brand recognition."""
import os
from pathlib import Path
root=Path(__file__).resolve().parent
os.chdir(root)
import torch,yaml,ultralytics
from ultralytics import YOLOE
torch.set_num_threads(2)
model=YOLOE(str(root/'yoloe-26s-seg.pt'))
with open(Path(ultralytics.__file__).parent/'cfg/datasets/coco.yaml') as f:
 names=list(yaml.safe_load(f)['names'].values())
names.append('beverage can')
model.set_classes(names)
model.save(str(root/'yoloe-26s-cans.pt'))
print('Prepared 80 existing categories + beverage can; no brand identity inferred.',flush=True)
