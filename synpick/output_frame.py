
"""
Methods for writing an output frame
"""

import stillleben as sl

from pathlib import Path
from PIL import Image

def write_frame(path : Path, idx : int, scene : sl.Scene, result : sl.RenderPassResult):

    # TODO: Augmentation?
    rgb = Image.fromarray(result.rgb()[:,:,:3].cpu().numpy())
    rgb.save(path / 'rgb' / f'{idx:06}.jpg')

    depth = (result.depth().cpu() * 1000.0).short().numpy()
    depth = Image.fromarray(depth)
    depth.save(path / 'depth' / f'{idx:06}.png')
