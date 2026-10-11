"""Reuse cached embeddings to retain only billiard balls and cue balls."""
from pathlib import Path
import torch
from ultralytics import YOLOE
root=Path(__file__).resolve().parent
torch.set_num_threads(2)
model=YOLOE(str(root/'yoloe-26s-billiards.pt'))
names=['billiard ball','cue ball']
indices=[next(k for k,v in model.names.items() if v==name) for name in names]
model.set_classes(names,model.model.pe[:,indices].clone())
model.save(str(root/'yoloe-26s-balls-only.pt'))
print('Saved ball-only checkpoint with 2 cached categories; no training or text encoder required.',flush=True)
