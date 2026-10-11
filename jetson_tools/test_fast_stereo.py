"""Headless local Fast-FoundationStereo smoke test on official sample images."""
import os
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'Fast-FoundationStereo'
sys.path.insert(0,str(ROOT))
import cv2,numpy as np,torch
from core.utils.utils import InputPadder
import core.foundation_stereo

torch.set_num_threads(2)
torch.set_grad_enabled(False)
model=torch.load(ROOT/'weights/20-30-48/model_best_bp2_serialize.pth',map_location='cpu',weights_only=False).cuda().eval()
model.args.max_disp=96
imgs=[]
for name in ['left','right']:
 im=cv2.cvtColor(cv2.imread(str(ROOT/f'demo_data/{name}.png')),cv2.COLOR_BGR2RGB)
 im=cv2.resize(im,(320,240))
 imgs.append(torch.from_numpy(im.copy()).cuda().float().permute(2,0,1)[None])
pad=InputPadder(imgs[0].shape,divis_by=32,force_square=False)
a,b=pad.pad(*imgs)
for i in range(3):
 t=time.monotonic()
 with torch.amp.autocast('cuda',dtype=torch.float16):
  out=model(a,b,iters=4,test_mode=True,optimize_build_volume='pytorch1')
 torch.cuda.synchronize()
 d=pad.unpad(out.float()).cpu().numpy().squeeze()
 print(json.dumps({'iteration':i,'seconds':time.monotonic()-t,'finite_fraction':float(np.isfinite(d).mean()),'median_disparity_px':float(np.median(d)), 'gpu_peak_MB':torch.cuda.max_memory_allocated()/1e6}),flush=True)
 np.save(ROOT/'output_docker/jetson_sample_disparity.npy',d)
