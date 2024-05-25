import json
import os
import time
from typing import List

import PIL.Image

from ..panda import Panda
from ..rotation_utils import vector2quat
from .rgbd import RGBD

# Go to a set of waypoints.
# These should get 4 samples from each coordinate, in unique combinations,
# by traversing the vertices of a cube.

def create_waypoints(x1, y1, z1, x2, y2, z2):

    waypoints = [
        # Cycle 1: back cube face
        [x1, y1, z1],
        [x1, y1, z2],
        [x1, y2, z2],
        [x1, y2, z1],
        [x1, y1, z1],

        # Cycle 2: front cube face
        [x2, y1, z1],
        [x2, y1, z2],
        [x2, y2, z2],
        [x2, y2, z1],
        [x2, y1, z1],
    ]

    return waypoints

def create_NxN_waypoints(X, Y, Z):
    for xi in range(len(X)):
        for yi in range(len(Y)):
            for zi in range(len(Z)):
                yield [X[xi], Y[yi], Z[zi]]

waypoints = {
    "front": create_waypoints(
        x1=0.4,
        x2=0.6,
        y1=-0.2,
        y2=0.2,
        z1=0.1,
        z2=0.3,
    ),
    "back_left": list(create_NxN_waypoints(
        X=[0.3, 0.4, 0.5],
        Y=[0.1, -0.1, -0.3],
        Z=[0.1, 0.2, 0.3],
    )),
    "back_right": create_waypoints(
        x1=0.3,
        x2=0.5,
        y1=-0.1,
        y2=0.3,
        z1=0.1,
        z2=0.3,
    ),
    "front_left": create_waypoints(
        x1=0.3,
        x2=0.5,
        y1=-0.2,
        y2=0.2,
        z1=0.1,
        z2=0.3,
    ),
}

def collect_trajectory(preset):
    calibration_out_dir = os.path.join(os.path.dirname(__file__), f"calibration_images_{preset}")

    robot = Panda("192.168.1.222")

    # "claw" parameter = what direction the palm of the claw faces
    # "right" parameter = what direction the side of the end effector with
    # the gray button should face
    accept_cube_rot = vector2quat([0, 1, 0], [0, 0, -1])
    vertical_rot = vector2quat([0, 0, -1], [0, -1, 0])

    # Wait to accept the cube.
    robot.move_to([0.4, 0.0, 0.1], accept_cube_rot, direct=True)

    if robot.gripper.get_state().width < 0.02:
        robot.stop_grasp()

    input("Place cube in claw.")

    # Hold the cube
    robot.start_grasp()

    camera_id = '000259521012' if 'front' in preset else '000243521012'
    rgbd = RGBD(camera_ids=[camera_id])

    imgs: List[PIL.Image.Image] = []
    eef_poses = []

    for i, waypoint in enumerate(waypoints[preset]):
        print(f"Moving to waypoint {i + 1} / {len(waypoints[preset])}: ", waypoint)
        robot.move_to(waypoint, vertical_rot, direct=True)

        time.sleep(2.0)

        (rgbs, pcds) = rgbd.capture()
        # measure the actual end-effector position
        eef_pose = robot.get_pos()[0]
        eef_poses.append(
            eef_pose.numpy().tolist()
        )

        imgs.append(PIL.Image.fromarray(rgbs[0]))

    robot.stop_grasp()

    # Now we can process the images
    # The apriltag locations will be offset in the +x direction by 1/2 * (cube size),
    # where (cube size) = 0.04m (in our case).
    os.makedirs(calibration_out_dir, exist_ok=True)

    for i, img in enumerate(imgs):
        img.save(os.path.join(calibration_out_dir, f"img_{i:02d}.png"))

    with open(os.path.join(calibration_out_dir, "eef_poses.json"), "w") as f:
        json.dump(eef_poses, f)

if __name__ == "__main__":
    preset = 'back_right'
    camera_id = '000259521012' if 'front' in preset else '000243521012'
    calibration_out_dir = os.path.join(os.path.dirname(__file__), f"calibration_images_{preset}")

    if os.path.exists(calibration_out_dir):
        print("calibration_images directory already exists. Please delete it first.")
        exit(1)
    else:
        collect_trajectory(preset)

    from .run_calibration import run_calibration
    from .camera_params import camera_params

    (rvec, rmat, tvec) = run_calibration(preset, **camera_params[camera_id])
    with open(os.path.join(os.path.dirname(__file__), f"extrinsics/{preset}_camera.json"), "w") as f:
        json.dump({
            "rvec": rvec.tolist(),
            "rotation_matrix": rmat.tolist(),
            "translation": tvec.tolist(),
        }, f)
