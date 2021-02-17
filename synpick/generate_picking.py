

from synpick.object_models import load_gripper, load_tote, load_meshes, OBJECT_NAMES
from synpick.scene import create_scene, CAMERA_POSES
from synpick.output import Writer
from synpick.gripper_sim import GripperSim
from synpick.picking_heuristic import postprocess_segmentation, visualize_detections, postprocess_with_depth, process_detections

from pathlib import Path
from typing import Optional
from contextlib import ExitStack
import random
import stillleben as sl
import torch
import json

from PIL import Image

def run(out : Path, start_index : int, ibl_path : Path, visualize : bool = False):

    meshes = load_meshes()
    mesh_pool = list(meshes)

    object_sizes = torch.stack(
        [torch.zeros(3)] + \
        [ mesh.bbox.size for mesh in meshes ]
    )
    object_weights = [0.0] + [ sl.Object(m).mass for m in meshes ]

    scene = create_scene(ibl_path)

    depth_scale = 0.1

    # Add meshes
    volume = 0
    while volume < 7.0 / 1000.0:
        mesh = random.choice(mesh_pool)
        mesh_pool.remove(mesh)

        obj = sl.Object(mesh)
        obj.instance_index = len(scene.objects)+1

        scene.add_object(obj)
        volume += obj.volume

    print('Dropping items...')
    scene.simulate_tabletop_scene()
    print('Done.\n')

    # Load gripper
    gripper = sl.Object(load_gripper())
    gripper.metallic = 0.01
    gripper.roughness = 0.9
    gripper.linear_velocity_limit = 0.06

    # Manipulation simulation
    gripper_pose = torch.eye(4)
    #gripper_pose[:3,3] = waypoints[0]
    gripper_pose[2,3] = 0.5
    sim = GripperSim(scene, gripper, gripper_pose)
    sim.set_spring_parameters(1000.0, 1.0, 30.0)

    renderer = sl.RenderPass()

    GRIPPER_VELOCITY = 0.1
    DT = 0.002
    STEPS_PER_FRAME = int(round((1.0 / 24) / DT))

    frame_idx = 0

    # Create an output writer for each camera pose
    writers = [ Writer(out / f'{start_index+i:06}') for i in range(len(CAMERA_POSES)) ]

    gripper_out_of_way = torch.eye(4)
    gripper_out_of_way[2,3] = 10.0

    # Generate sequence!
    with ExitStack() as stack:
        for writer in writers:
            stack.enter_context(writer)

        while True:
            # Render once from above
            scene.set_camera_pose(CAMERA_POSES[0])

            # without gripper
            saved_gripper_pose = gripper.pose()
            gripper.set_pose(gripper_out_of_way)

            result = renderer.render(scene)

            segmentation = result.class_index()[:,:,0]
            detections = postprocess_segmentation(
                segmentation=segmentation,
                confidence=torch.ones_like(segmentation),
                classes=OBJECT_NAMES,
                object_sizes=object_sizes,
                object_weights=object_weights
            )
            postprocess_with_depth(
                detections=detections,
                segmentation=segmentation,
                confidence=torch.ones_like(segmentation),
                classes=OBJECT_NAMES,
                cloud=result.cam_coordinates()[:,:,:3],
                object_sizes=object_sizes,
            )

            vis = visualize_detections(result.rgb()[:,:,:3], detections, True)
            Image.fromarray(result.rgb()[:,:,:3].cpu().numpy()).save('/tmp/rgb.png')
            Image.fromarray(vis.numpy()).save('/tmp/vis.png')

            process_detections(detections)

            print(detections)

            # restore gripper pose
            gripper.set_pose(saved_gripper_pose)

            return

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
