import os
import time
from typing import List

import PIL.Image
from compete_and_select.panda import Panda
from compete_and_select.rotation_utils import vector2quat
from compete_and_select.perception.rgbd import RGBD

# Go to a set of waypoints.
# These should get 4 samples from each coordinate, in unique combinations,
# by traversing the vertices of a cube.

x1 = 0.4
x2 = 0.6
y1 = -0.2
y2 = 0.2
z1 = 0.1
z2 = 0.3

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

calibration_out_dir = os.path.join(os.path.dirname(__file__), "calibration_images")

def collect_trajectory():
    robot = Panda("192.168.1.222")

    # "claw" parameter = what direction the palm of the claw faces
    # "right" parameter = what direction the side of the end effector with
    # the gray button should face
    accept_cube_rot = vector2quat([0, 1, 0], [0, 0, -1])
    vertical_rot = vector2quat([0, 0, -1], [0, -1, 0])

    # Wait to accept the cube.
    robot.move_to([0.4, 0.0, 0.1], accept_cube_rot, direct=True)

    input("Place cube in claw.")

    # Hold the cube
    robot.start_grasp()

    # Rotate to vertical
    robot.move_to([0.4, 0.0, 0.1], vertical_rot, direct=True)

    rgbd = RGBD(num_cameras=1)

    imgs: List[PIL.Image.Image] = []

    for i, waypoint in enumerate(waypoints):
        print(f"Moving to waypoint {i + 1} / 10: ", waypoint)
        robot.move_to(waypoint, vertical_rot, direct=True)

        time.sleep(2.0)

        (rgbs, pcds) = rgbd.capture()

        imgs.append(PIL.Image.fromarray(rgbs[0]))

    robot.stop_grasp()

    # Now we can process the images
    # The apriltag locations will be offset in the +x direction by 1/2 * (cube size),
    # where (cube size) = 0.04m (in our case).
    os.makedirs(calibration_out_dir, exist_ok=True)

    for i, img in enumerate(imgs):
        img.save(os.path.join(calibration_out_dir, f"img_{i:02d}.png"))

if __name__ == "__main__":
    if os.path.exists(calibration_out_dir):
        print("calibration_images directory already exists. Please delete it first.")
        exit(1)
    else:
        collect_trajectory()
