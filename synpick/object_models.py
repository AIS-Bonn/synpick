
"""
Load the object models.
"""

import torch
import stillleben as sl
from pathlib import Path

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
