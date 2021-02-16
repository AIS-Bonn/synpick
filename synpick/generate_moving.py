
from synpick.object_models import load_gripper, load_tote, load_meshes
from synpick.scene import create_scene, CAMERA_POSES
from synpick.output import Writer

from pathlib import Path
from typing import Optional
from contextlib import ExitStack
import random
import stillleben as sl
import torch
import json

def run(out : Path, start_index : int, ibl_path : Path, visualize : bool = False):

    meshes = load_meshes()

    scene = create_scene(ibl_path)

    depth_scale = 0.1

    # Add meshes
    volume = 0
    while volume < 7.0 / 1000.0:
        mesh = random.choice(meshes)

        obj = sl.Object(mesh)
        obj.instance_index = len(scene.objects)+1

        scene.add_object(obj)
        volume += obj.volume

    print('Dropping items...')
    scene.simulate_tabletop_scene()
    print('Done.\n')

    tote_size = scene.objects[0].mesh.bbox.size
    corner = 0.75 * (tote_size/2)
    corners = []
    for ix in (-1, 1):
        for iy in (-1, 1):
            corners.append(torch.tensor([ix, iy, 0]) * corner + torch.tensor([0.0,0.0,0.02]))

    waypoints = random.choices(corners, k=8)

    # Load gripper
    gripper = sl.Object(load_gripper())
    gripper.metallic = 0.01
    gripper.roughness = 0.9
    gripper.linear_velocity_limit = 0.06

    # Manipulation simulation
    gripper_pose = torch.eye(4)
    gripper_pose[:3,3] = waypoints[0]
    gripper_pose[2,3] = 0.5
    sim = sl.ManipulationSim(scene, gripper, gripper_pose)
    sim.set_spring_parameters(1000.0, 1.0, 30.0)

    renderer = sl.RenderPass()

    GRIPPER_VELOCITY = 0.1
    DT = 0.002
    STEPS_PER_FRAME = int(round((1.0 / 24) / DT))

    frame_idx = 0

    # Create an output writer for each camera pose
    writers = [ Writer(out / f'{start_index+i:06}') for i in range(len(CAMERA_POSES)) ]

    # Generate sequence!
    with ExitStack() as stack:
        for writer in writers:
            stack.enter_context(writer)

        for wp in waypoints:
            while True:
                delta = wp - gripper_pose[:3,3]

                dn = delta.norm()

                if dn < GRIPPER_VELOCITY*DT + 0.001:
                    break

                delta = delta / dn * GRIPPER_VELOCITY*DT

                gripper_pose[:3,3] += delta

                sim.step(gripper_pose, DT)

                if frame_idx % STEPS_PER_FRAME == 0:
                    for writer, camera_pose in zip(writers, CAMERA_POSES):
                        scene.set_camera_pose(camera_pose)
                        result = renderer.render(scene)
                        writer.write_frame(scene, result)

                frame_idx += 1

    print('Finished')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ibl', metavar='PATH', type=str, required=True,
        help='Directory containing sIBL maps')
    parser.add_argument('--out', metavar='PATH', type=str, required=True,
        help='Output frame directory (should not exist)')
    parser.add_argument('--base', metavar='N', type=int, required=True,
        help='Number of first output sequence')
    parser.add_argument('--viewer', action='store_true')

    args = parser.parse_args()

    sl.init_cuda()
    run(out=Path(args.out), start_index=args.base, ibl_path=Path(args.ibl), visualize=args.viewer)
