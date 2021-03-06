
"""
Setup scene parameters
"""

import stillleben as sl
import torch
import math
from pathlib import Path

from .object_models import load_tote

RESOLUTION = (1920, 1080)
FOV_X = 20.0 * math.pi / 180.0

#RESOLUTION = (640,480)
#FOV_X = 20.0 * math.pi / 180.0

TOTE_DIM = torch.tensor([0.615 + 0.1, 0.373 + 0.2, 0.2])
CAMERA_POSITIONS = [
    torch.tensor([0, 0, 2.0]),
    #torch.tensor([-TOTE_DIM[0]/2, 0, 1.2]),
    #torch.tensor([TOTE_DIM[0]/2,  0, 1.2]),
    torch.tensor([0,  TOTE_DIM[1], 2.0]),
    torch.tensor([0, -TOTE_DIM[1], 2.0])
]

def camPoseFromPosition(p):
    lookAt = torch.tensor([0.0, 0.0, 0.8*TOTE_DIM[2]])

    up = -p
    up[2] = 0.0

    if up.norm() < 1e-3:
        up = torch.tensor([0.0, 1.0, 0.0])

    zAxis = lookAt - p
    zAxis /= zAxis.norm()

    xAxis = torch.cross(zAxis, up)
    xAxis /= xAxis.norm()

    yAxis = torch.cross(zAxis, xAxis)
    yAxis /= yAxis.norm()

    T = torch.eye(4)
    T[:3,0] = xAxis
    T[:3,1] = yAxis
    T[:3,2] = zAxis
    T[:3,3] = p

    return T

CAMERA_POSES = [ camPoseFromPosition(p) for p in CAMERA_POSITIONS ]

def create_scene(ibl_path : Path):
    # Create a scene with specified intrinsics
    scene = sl.Scene(RESOLUTION)
    scene.set_camera_hfov(FOV_X)

    # Add red plastic tote
    tote_obj = sl.Object(load_tote())
    tote_obj.static = True
    scene.add_object(tote_obj)
    tote_obj.instance_index = 0

    scene.choose_random_camera_pose()

    # Setup lighting
    scene.light_map = sl.LightMap(ibl_path)

    return scene

def load_light_maps():
    return [
        sl.LightMap(p) for p in Path('external_data/ibl').glob('*/*.ibl')
    ]
