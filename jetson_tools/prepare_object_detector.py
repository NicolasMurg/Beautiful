"""Cache YOLOE category prompts; no custom training or brand identification."""
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
names.extend(['beverage can','ball','billiard ball','cue ball'])
model.set_classes(names)
model.save(str(root/'yoloe-26s-objects.pt'))
print('Prepared 84 model categories including beverage can, ball, billiard ball and cue ball.',flush=True)
