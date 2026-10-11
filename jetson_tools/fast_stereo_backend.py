"""Optional local NVlabs stereo matcher; input is already rectified BGR."""
from pathlib import Path
import sys


class FastStereoMatcher:
    def __init__(self, device='0', max_disp=96, iters=4):
        import torch
        root = Path(__file__).resolve().parent / 'Fast-FoundationStereo'
        sys.path.insert(0, str(root))
        import core.foundation_stereo  # register checkpoint classes
        from core.utils.utils import InputPadder
        self.torch, self.padder = torch, InputPadder
        self.device = torch.device(f'cuda:{device}')
        self.model = torch.load(root / 'weights/20-30-48/model_best_bp2_serialize.pth',
                                map_location='cpu', weights_only=False).to(self.device).eval()
        self.model.args.max_disp = max_disp
        self.iters = iters

    def predict(self, left, right):
        import cv2
        import numpy as np
        torch = self.torch
        images = [torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).copy())
                  .to(self.device).float().permute(2, 0, 1)[None] for im in (left, right)]
        pad = self.padder(images[0].shape, divis_by=32, force_square=False)
        a, b = pad.pad(*images)
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            result = self.model(a, b, iters=self.iters, test_mode=True,
                                optimize_build_volume='pytorch1')
        disparity = pad.unpad(result.float()).cpu().numpy().squeeze().copy()
        disparity[~np.isfinite(disparity) | (disparity <= 0)] = np.nan
        return disparity

    def both(self, left, right):
        # Mirror and swap to make right-to-left correspondence positive for
        # the network, then restore the right image's coordinates/sign.
        dl = self.predict(left, right)
        dr = -self.predict(right[:, ::-1].copy(), left[:, ::-1].copy())[:, ::-1].copy()
        return dl, dr
