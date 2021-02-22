
"""
Methods for writing an output frame
"""

import stillleben as sl
import torch
import time

from pathlib import Path

class Writer(object):
    def __init__(self, path : Path):
        self.path = path
        self.idx = 0
        self.depth_scale = 10000.0 # depth [m] = pixel / depth_scale
        self.saver = sl.ImageSaver()

        # Create output directory
        path.mkdir(parents=True)

        (path / 'rgb').mkdir()
        (path / 'mask_visib').mkdir()
        (path / 'depth').mkdir()

        self.camera_file = open(path / 'scene_camera.json', 'w')
        self.camera_file.write('{\n')

        self.gt_file = open(path / 'scene_gt.json', 'w')
        self.gt_file.write('{\n')

        self.log_file = open(path / 'log.txt', 'w')

    def __enter__(self):
        self.saver.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        # Finish camera_file
        self.camera_file.write('\n}')
        self.camera_file.close()

        # Finish gt_file
        self.gt_file.write('\n}')
        self.gt_file.close()

        # Finish log file
        self.log_file.close()

        self.saver.__exit__(type, value, traceback)

    @staticmethod
    def intrinsicMatrixFromProjection(proj : torch.tensor, W : int, H : int):
        far = -proj[2,3] / (proj[2,2] - 1.0)
        near = (proj[2,2] - 1.0) / (proj[2,2] + 1.0) * far
        left = -near * (proj[0,2]+1) / proj[0,0]
        right = -near * (proj[0,2]-1) / proj[0,0]
        bottom = -near * (proj[1,2]-1) / proj[1,1]
        top = -near * (proj[1,2]+1) / proj[1,1]

        eps = 2.2204460492503131e-16

        if abs(left-right) < eps:
            cx = W * 0.5
        else:
            cx = (left * W) / (left - right)

        if abs(top-bottom) < eps:
            cy = H * 0.5
        else:
            cy = (top * H) / (top - bottom)

        fx = -near * cx / left
        fy = -near * cy / top

        return torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ])

    def write_log(self, *args, **kwargs):
        self.log_file.write(f'{self.idx:06}: ')
        print(*args, **kwargs, file=self.log_file)

    def write_scene_data(self, scene : sl.Scene):
        with open(self.path / 'scene.sl', 'w') as f:
            f.write(scene.serialize())

    def write_frame(self, scene : sl.Scene, result : sl.RenderPassResult):

        # TODO: Augmentation?
        #t0 = time.time()
        rgb = result.rgb()[:,:,:3].cpu().contiguous()
        #t1 = time.time()

        #print(f'RGB: {t1-t0}')

        self.saver.save(
            rgb,
            str(self.path / 'rgb' / f'{self.idx:06}.jpg')
        )

        #t0 = time.time()
        depth = (result.depth() * self.depth_scale).short().cpu().contiguous()
        #t1 = time.time()

        #print(f'Depth: {t1-t0}')


        self.saver.save(
            depth,
            str(self.path / 'depth' / f'{self.idx:06}.png')
        )

        # Figure out cam_K
        P = scene.projection_matrix()
        W,H = scene.viewport

        cam_K = Writer.intrinsicMatrixFromProjection(P, W, H)

        world_in_camera = torch.inverse(scene.camera_pose())
        cam_R_w2c = world_in_camera[:3,:3].contiguous()
        cam_t_w2c = world_in_camera[:3,3] * 1000.0 # millimeters, of course.

        # Write scene_camera.json
        if self.idx != 0:
            self.camera_file.write(',\n')
        self.camera_file.write(f'  "{self.idx}": {{"cam_K": {cam_K.view(-1).tolist()}, "depth_scale": {1.0 / (self.depth_scale / 1000.0)}, "cam_R_w2c": {cam_R_w2c.view(-1).tolist()}, "cam_t_w2c": {cam_t_w2c.tolist()}}}')

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
