
"""
Setup scene parameters
"""

import stillleben as sl
from pathlib import Path

from .object_models import load_tote

RESOLUTION = (640, 480)
INTRINSICS = (1066.778, 1067.487, 312.9869, 241.3109)

def create_scene(ibl_path : Path):
    # Create a scene with specified intrinsics
    scene = sl.Scene(RESOLUTION)
    scene.set_camera_intrinsics(*INTRINSICS)

    # Add red plastic tote
    tote_obj = sl.Object(load_tote())
    tote_obj.static = True
    scene.add_object(tote_obj)

    scene.choose_random_camera_pose()

    # Setup lighting
    scene.light_map = sl.LightMap(ibl_path)

    return scene
