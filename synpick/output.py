
"""
Methods for writing an output frame
"""

import stillleben as sl
import torch

from pathlib import Path
from PIL import Image

class Writer(object):
    def __init__(self, path : Path):
        self.path = path
        self.idx = 0
        self.depth_scale = 10000.0 # depth [m] = pixel / depth_scale

        # Create output directory
        path.mkdir(parents=True)

        (path / 'rgb').mkdir()
        (path / 'mask_visib').mkdir()
        (path / 'depth').mkdir()

        self.camera_file = open(path / 'scene_camera.json', 'w')
        self.camera_file.write('{\n')

        self.gt_file = open(path / 'scene_gt.json', 'w')
        self.gt_file.write('{\n')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # Finish camera_file
        self.camera_file.write('\n}')
        self.camera_file.close()

        # Finish gt_file
        self.gt_file.write('\n}')
        self.gt_file.close()

    def write_frame(self, scene : sl.Scene, result : sl.RenderPassResult):

        # TODO: Augmentation?
        rgb = Image.fromarray(result.rgb()[:,:,:3].cpu().numpy())
        rgb.save(self.path / 'rgb' / f'{self.idx:06}.jpg')

        depth = (result.depth().cpu() * self.depth_scale).short().numpy()
        depth = Image.fromarray(depth)
        depth.save(self.path / 'depth' / f'{self.idx:06}.png')

        # Figure out cam_K
        P = scene.projection_matrix()
        W,H = scene.viewport

        glToCV = torch.tensor([
            [W/2.0, 0.0, W/2.0],
            [0.0, H/2.0, H/2.0],
            [0.0, 0.0, 1.0]
        ])
        cam_K = glToCV @ P[:3,:3]

        world_in_camera = torch.inverse(scene.camera_pose())

        # Write scene_camera.json
        if self.idx != 0:
            self.camera_file.write(',\n')
        self.camera_file.write(f'  "{self.idx}": {{"cam_K": {cam_K.view(-1).tolist()}, "depth_scale": {1.0 / (self.depth_scale / 1000.0)}}}')

        # Write scene_gt.json
        if self.idx != 0:
            self.gt_file.write(',\n\n')

        def gt(o):
            T = world_in_camera @ o.pose()

            cam_R_m2c = T[:3,:3].contiguous()
            cam_t_m2c = T[:3,3] * 1000.0 # millimeters, of course.

            return f'{{"cam_R_m2c": {cam_R_m2c.view(-1).tolist()}, "cam_t_m2c": {cam_t_m2c.tolist()}, "obj_id": {o.mesh.class_index}}}'

        formatted_gt = ",\n".join([ gt(o) for o in scene.objects if o.mesh.class_index > 0 ])
        self.gt_file.write(f'  "{self.idx}": [\n    {formatted_gt}]')

        self.idx += 1
