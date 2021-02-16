#!/usr/bin/env blender -b -P

import bpy
import os

path = 'external_data/ycbv_models/models_fine'  # set this path

for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith('.ply') :
            mesh_file = os.path.join(path, f)
            out_file = os.path.splitext(mesh_file)[0] + ".glb"

            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()

            bpy.ops.import_mesh.ply(filepath=mesh_file) # change this line

            bpy.ops.object.select_all(action='SELECT')

            bpy.ops.export_scene.gltf(filepath=out_file, export_format='GLB', export_apply=True)
