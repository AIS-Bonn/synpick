
"""
Load the object models.
"""

import torch
import stillleben as sl
from pathlib import Path

OBJECT_NAMES = [
    'box',

    '002_master_chef_can',
    '003_cracker_box',
    '004_sugar_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '007_tuna_fish_can',
    '008_pudding_box',
    '009_gelatin_box',
    '010_potted_meat_can',
    '011_banana',
    '019_pitcher_base',
    '021_bleach_cleanser',
    '024_bowl',
    '025_mug',
    '035_power_drill',
    '036_wood_block',
    '037_scissors',
    '040_large_marker',
    '051_large_clamp',
    '052_extra_large_clamp',
    '061_foam_brick',
]

def load_meshes():
    path = Path('external_data/ycbv_models/models_fine')

    mesh_files = [ path / f'obj_{i+1:06}.ply' for i in range(21) ]

    meshes = sl.Mesh.load_threaded(mesh_files, flags=sl.Mesh.Flag.PHYSICS_FORCE_CONVEX_HULL)

    # Edit mesh properties
    for i, mesh in enumerate(meshes):
        # 1) BOP models use millimeter units for some strange reason. sl is
        # completely metric, so scale accordingly.
        pt = torch.eye(4)
        pt[:3,:3] *= 0.001
        mesh.pretransform = pt

        # 2) Set up class indices properly
        mesh.class_index = i + 1

    return meshes

def load_tote():
    tote = sl.Mesh('meshes/tote.glb')
    tote.class_index = 0
    return tote

def load_gripper():
    gripper = sl.Mesh('meshes/gripper.glb', flags=sl.Mesh.Flag.PHYSICS_FORCE_CONVEX_HULL)
    gripper.class_index = 0
    return gripper

if __name__ == "__main__":
    sl.init()

    meshes = load_meshes()

    for mesh in meshes:
        obj = sl.Object(mesh)

        print(f" - Mesh {mesh.filename}:")

        bbox = mesh.bbox.size
        print(f"   BBox {bbox[0]:.2f}x{bbox[1]:.2f}x{bbox[2]:.2f}, Mass {obj.mass:.3f}kg")
