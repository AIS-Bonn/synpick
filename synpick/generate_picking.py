

from synpick.object_models import load_gripper_base, load_gripper_cup, load_tote, load_meshes, OBJECT_NAMES, get_object, OBJECT_INFO
from synpick.scene import create_scene, CAMERA_POSES
from synpick.output import Writer
from synpick.gripper_sim import GripperSim
from synpick.picking_heuristic import postprocess_segmentation, visualize_detections, postprocess_with_depth, process_detections
from synpick.color_classes import Colorizer

from pathlib import Path
from typing import Optional
from contextlib import ExitStack
import random
import stillleben as sl
import torch
import json

from PIL import Image

def compute_grasp_frame(position, normal):
    zAxis = normal
    xAxis = torch.cross(torch.tensor([1.0,0.0,0.0]), zAxis)
    xAxis /= xAxis.norm()

    yAxis = torch.cross(zAxis, xAxis)
    yAxis /= yAxis.norm()

    T = torch.eye(4)
    T[:3,0] = xAxis
    T[:3,1] = yAxis
    T[:3,2] = zAxis
    T[:3,3] = position

    return T

def translation(x, y, z):
    T = torch.eye(4)
    T[:3,3] = torch.tensor([x,y,z])
    return T

def run(out : Path, start_index : int, ibl_path : Path, visualize : bool = False, bad : bool = False):

    meshes = load_meshes()
    mesh_pool = list(range(len(meshes)))

    object_sizes = torch.stack(
        [torch.zeros(3)] + \
        [ mesh.bbox.size for mesh in meshes ]
    )
    object_weights = [0.0] + [ sl.Object(m).mass for m in meshes ]

    scene = create_scene(ibl_path)

    depth_scale = 0.1

    # Add meshes
    volume = 0
    obj_idx = 1
    while volume < 7.0 / 1000.0:
        mesh_idx = random.choice(mesh_pool)
        mesh_pool.remove(mesh_idx)

        obj = get_object(meshes[mesh_idx], OBJECT_INFO[mesh_idx+1])

        scene.add_object(obj)

        obj.instance_index = obj_idx
        obj_idx += 1

        volume += obj.volume
        print(volume)

    print('Dropping items...')
    scene.simulate_tabletop_scene()
    print('Done.\n')

    # Load gripper
    gripper = sl.Object(load_gripper_base())
    gripper.metallic = 0.01
    gripper.roughness = 0.9

    scene.add_object(gripper)
    gripper.instance_index = 0

    gripper_cup = sl.Object(load_gripper_cup())
    gripper_cup.metallic = 0.01
    gripper_cup.roughness = 0.9

    scene.add_object(gripper_cup)
    gripper_cup.instance_index = 0

    # Manipulation simulation
    gripper_pose = torch.eye(4)
    #gripper_pose[:3,3] = waypoints[0]
    gripper_pose[2,3] = 0.5
    sim = GripperSim(scene, gripper, gripper_cup, gripper_pose)
    sim.set_spring_parameters(2500.0, 200.0, 200.0)

    renderer = sl.RenderPass()

    GRIPPER_VELOCITY = 0.5
    gripper.linear_velocity_limit = 2.0 * GRIPPER_VELOCITY
    DT = 0.002
    STEPS_PER_FRAME = int(round((1.0 / 15) / DT))

    frame_idx = 0
    failed_picks = 0

    # Create an output writer for each camera pose
    writers = [ Writer(out / f'{start_index+i:06}') for i in range(len(CAMERA_POSES)) ]

    gripper_out_of_way = translation(0.0, 0.0, 10.0)

    if visualize:
        viewer = sl.Viewer(scene)
    else:
        viewer = None

    # Define helper functions
    def log(*args, **kwargs):
        for writer in writers:
            writer.write_log(*args, **kwargs)

        print(*args, **kwargs)

    def move_gripper_to(pos, vel=GRIPPER_VELOCITY):
        nonlocal frame_idx
        nonlocal gripper_pose

        log(f'Moving from {gripper_pose[:3,3].tolist()} to {pos.tolist()}')

        while True:
            delta = pos - gripper_pose[:3,3]

            dn = delta.norm()

            if dn < vel*DT + 0.001:
                break

            delta = delta / dn * vel*DT

            gripper_pose[:3,3] += delta

            sim.step(gripper_pose[:3,3], DT)

            # Remove any objects that have fallen out
            objects = list(scene.objects)
            for obj in objects:
                if obj.pose()[2,3] < -0.1:
                    scene.remove_object(obj)

            if frame_idx % STEPS_PER_FRAME == 0:
                for writer, camera_pose in zip(writers, CAMERA_POSES):
                    scene.set_camera_pose(camera_pose)
                    result = renderer.render(scene)
                    writer.write_frame(scene, result)

                    if viewer:
                        viewer.draw_frame()

            frame_idx += 1

    last_pick_failed = False

    if dump_perception:
        colorizer = Colorizer(len(OBJECT_INFO))

    # Generate sequence!
    with ExitStack() as stack:

        # Write once-per-sequence data
        for writer, camera_pose in zip(writers, CAMERA_POSES):
            stack.enter_context(writer)

            scene.set_camera_pose(camera_pose)
            writer.write_scene_data(scene)

        # Execute!
        while failed_picks < 4:
            # Render once from above
            scene.set_camera_pose(CAMERA_POSES[0])

            # without gripper
            saved_gripper_pose = gripper.pose()
            gripper.set_pose(gripper_out_of_way)
            gripper_cup.set_pose(gripper_out_of_way)

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

            if dump_perception:
                vis = visualize_detections(result.rgb()[:,:,:3], detections, grasps=True, labels=False)
                Image.fromarray(result.rgb()[:,:,:3].cpu().numpy()).save('rgb.jpg')

                for det in detections:
                    colorizer.colors[det.class_id] = det.color

                colorized = colorizer.colorize(result.class_index()[:,:,0].long()).cpu()
                Image.fromarray(colorized.permute(1,2,0).numpy()).save('segmentation.png')
                Image.fromarray(vis.numpy()).save('perception.png')

                with open('clutter.dot', 'w') as f:
                    f.write('digraph {\n')

                    for i, det in enumerate(detections):
                        f.write(f'v{det.name} [ label="{i}" color="#{det.color[0]:02X}{det.color[1]:02X}{det.color[2]:02X}" ];\n')

                    f.write(f'\n')

                    for det in detections:
                        for above in det.objects_above:
                            f.write(f'v{above} -> v{det.name};\n')

                    f.write('}\n')

                import sys
                sys.exit(0)

            log(f'Items found by perception:')
            for det in detections:
                log(f' - {det.name}')

            if bad and last_pick_failed:
                log('Random pick!')
                random.shuffle(detections)
            else:
                log('Heuristic pick!')
                process_detections(detections, return_bad_ones=bad)

            if len(detections) == 0:
                log("Empty!")
                break

            objectName = detections[0].name
            goalPixel = detections[0].suction_point
            log(f"I'm going to pick {objectName} at {goalPixel.tolist()}")

            coord = result.cam_coordinates()[goalPixel[1], goalPixel[0]].cpu()

            # Smooth normal in a small window
            S = 10
            norm_miny = max(0, goalPixel[1] - S)
            norm_maxy = min(scene.viewport[1]-1, goalPixel[1] + S)
            norm_minx = max(0, goalPixel[0] - S)
            norm_maxx = min(scene.viewport[0]-1, goalPixel[0] + S)
            normal = result.normals()[norm_miny:norm_maxy, norm_minx:norm_maxx, :3].mean(dim=0).mean(dim=0).cpu()
            log(f"Camera-space coords: {coord.tolist()}, normal: {normal.tolist()}")

            normal = normal / normal.norm()

            coord = scene.camera_pose() @ coord
            normal = scene.camera_pose()[:3,:3] @ normal

            log(f"World-space coords: {coord.tolist()}, normal: {normal.tolist()}")

            graspFrame = compute_grasp_frame(coord[:3], normal)

            start = graspFrame @ translation(0.0, 0.0, 0.4)
            close = graspFrame @ translation(0.0, 0.0, 0.05)
            above = translation(0.0, 0.0, 1.0) @ graspFrame
            away = translation(1.0, 0.0, 1.0) @ graspFrame
            gripper_pose[:] = above

            # 1) MOVEMENT: from pose above to grasp frame
            sim.prepare_grasp(normal, above[:3,3])

            move_gripper_to(close[:3,3], vel=1.0)
            move_gripper_to(graspFrame[:3,3], vel=0.1)

            # 2) SUCTION
            log(f"Arrived at object!")
            sim.enable_suction(40.0, 0.2)

            # 3) MOVEMENT: Back out
            move_gripper_to(above[:3,3])
            #move_gripper_to(away[:3,3])

            log(f"Back above")

            got_objects = sim.disable_suction()

            if len(got_objects) == 0:
                log(f"Grasp failed, got no object!")
                failed_picks += 1
                last_pick_failed = True
                continue

            last_pick_failed = False
            if bad:
                failed_picks = 0

            log(f"I got:")
            for obj in got_objects:
                log(f" - {OBJECT_NAMES[obj.mesh.class_index]} (ID {obj.mesh.class_index})")
                scene.remove_object(obj)

        log('Finished')

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
    parser.add_argument('--bad', action='store_true')
    parser.add_argument('--dump-perception', action='store_true')

    args = parser.parse_args()

    if torch.cuda.is_available():
        sl.init_cuda()
    else:
        sl.init()

    run(
        out=Path(args.out), start_index=args.base, ibl_path=Path(args.ibl),
        visualize=args.viewer, bad=args.bad, dump_perception=args.dump_perception
    )
