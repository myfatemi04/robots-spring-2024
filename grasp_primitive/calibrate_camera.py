import os
from typing import List

import PIL.Image
from compete_and_select.panda import Panda
from compete_and_select.rotation_utils import vector2quat
from compete_and_select.perception.rgbd import RGBD

robot = Panda("192.168.1.222")

# "claw" parameter = what direction the palm of the claw faces
# "right" parameter = what direction the side of the end effector with
# the gray button should face
accept_cube_rot = vector2quat([0, 1, 0], [0, 0, -1])
vertical_rot = vector2quat([0, 0, -1], [0, -1, 0])

# Wait to accept the cube.
robot.move_to([0.4, 0.0, 0.1], accept_cube_rot)

input("Place cube in claw.")

# Hold the cube
robot.start_grasp()

# Rotate to vertical
robot.move_to([0.4, 0.0, 0.1], vertical_rot)

# Go to a set of waypoints.
# These should get 4 samples from each coordinate, in unique combinations,
# by traversing the vertices of a cube.
waypoints = [
    # Cycle 1: back cube face
    [0.3, 0.0, 0.1],
    [0.3, 0.0, 0.2],
    [0.3, 0.1, 0.2],
    [0.3, 0.1, 0.1],
    [0.3, 0.0, 0.1],

    # Cycle 2: front cube face
    [0.5, 0.0, 0.1],
    [0.5, 0.0, 0.2],
    [0.5, 0.1, 0.2],
    [0.5, 0.1, 0.1],
    [0.5, 0.0, 0.1],
]

rgbd = RGBD()

imgs: List[PIL.Image.Image] = []

for waypoint in waypoints:
    robot.move_to(waypoint, vertical_rot)
    (rgbs, pcds) = rgbd.capture()

    imgs.append(PIL.Image.fromarray(rgbs[0]))

# Now we can process the images
calibration_out_dir = os.path.join(os.path.dirname(__file__), "calibration_images")

# The apriltag locations will be offset in the +x direction by 1/2 * (cube size),
# where (cube size) = 0.04m (in our case).

for i, img in enumerate(imgs):
    img.save(os.path.join(calibration_out_dir, f"img_{i:02d}.png"))
