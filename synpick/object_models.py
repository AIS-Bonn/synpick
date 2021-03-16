
"""
Load the object models.
"""

import torch
import stillleben as sl
from pathlib import Path

from collections import namedtuple

FLAG_CONCAVE = (1 << 0)

ObjectInfo = namedtuple('ObjectInfo', ['name', 'weight', 'flags', 'metallic', 'roughness'])

# source: http://www.ycbbenchmarks.com/wp-content/uploads/2015/09/object-list-Sheet1.pdf

OBJECT_INFO = [
    ObjectInfo('box', 1.0, 0, 0.1, 0.5),

    ObjectInfo('002_master_chef_can',    0.414,         0,             0.6, 0.2),
    ObjectInfo('003_cracker_box',        0.411,         0,             0.1, 0.5),
    ObjectInfo('004_sugar_box',          0.514,         0,             0.1, 0.5),
    ObjectInfo('005_tomato_soup_can',    0.349,         0,             0.1, 0.5),
    ObjectInfo('006_mustard_bottle',     0.603,         0,             0.3, 0.5),
    ObjectInfo('007_tuna_fish_can',      0.171,         0,             0.6, 0.2),
    ObjectInfo('008_pudding_box',        0.187,         0,             0.1, 0.5),
    ObjectInfo('009_gelatin_box',        0.097,         0,             0.1, 0.5),
    ObjectInfo('010_potted_meat_can',    0.370,         0,             0.6, 0.3),
    ObjectInfo('011_banana',             0.066,         0,             0.3, 0.3),
    ObjectInfo('019_pitcher_base',       0.178 + 0.066, 0,             0.1, 0.5),
    ObjectInfo('021_bleach_cleanser',    1.131,         0,             0.1, 0.5),
    ObjectInfo('024_bowl',               0.147,         FLAG_CONCAVE,  0.6, 0.3),
    ObjectInfo('025_mug',                0.118,         FLAG_CONCAVE,  0.6, 0.3),
    ObjectInfo('035_power_drill',        0.895,         FLAG_CONCAVE,  0.1, 0.6),
    ObjectInfo('036_wood_block',         0.729,         0,             0.3, 0.5),
    ObjectInfo('037_scissors',           0.082,         0,             0.1, 0.5),
    ObjectInfo('040_large_marker',       0.016,         0,             0.1, 0.5),
    ObjectInfo('051_large_clamp',        0.125,         0,             0.1, 0.5),
    ObjectInfo('052_extra_large_clamp',  0.202,         0,             0.1, 0.5),
    ObjectInfo('061_foam_brick',         0.028,         0,             0.1, 0.7),
]

OBJECT_NAMES = [ obj.name for obj in OBJECT_INFO ]
SYNPICK_DIR = Path(__file__).parent.parent.absolute()

def mesh_flags(info : ObjectInfo):
    if info.flags >= FLAG_CONCAVE:
        return sl.Mesh.Flag.NONE
    else:
        return sl.Mesh.Flag.PHYSICS_FORCE_CONVEX_HULL

def load_meshes():
    path = SYNPICK_DIR / Path('external_data/ycbv_models/models_fine')

    mesh_files = [ path / f'obj_{i+1:06}.ply' for i in range(21) ]

    flags = [ mesh_flags(info) for info in OBJECT_INFO[1:] ]

    meshes = sl.Mesh.load_threaded(mesh_files, flags=flags)

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

def get_object(mesh : sl.Mesh, objectInfo : ObjectInfo):
    print(mesh.filename, objectInfo.name)
    obj = sl.Object(mesh)

    obj.mass = objectInfo.weight
    obj.metallic = objectInfo.metallic
    obj.roughness = objectInfo.roughness

    return obj

def load_tote():
    tote = sl.Mesh(SYNPICK_DIR / 'meshes/tote.glb')
    tote.class_index = 0
    return tote

def load_gripper():
    gripper = sl.Mesh(SYNPICK_DIR / 'meshes/gripper.glb', flags=sl.Mesh.Flag.PHYSICS_FORCE_CONVEX_HULL)
    gripper.class_index = 0
    return gripper

def load_gripper_base():
    gripper = sl.Mesh(SYNPICK_DIR / 'meshes/gripper_base.glb', flags=sl.Mesh.Flag.PHYSICS_FORCE_CONVEX_HULL)
    gripper.class_index = 0
    return gripper

def load_gripper_cup():
    gripper = sl.Mesh(SYNPICK_DIR / 'meshes/gripper_cup.glb', flags=sl.Mesh.Flag.PHYSICS_FORCE_CONVEX_HULL)
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
