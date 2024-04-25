import numpy as np
from panda import Panda
from rgbd import RGBD
from scipy.spatial.transform import Rotation


def vector2quat(claw, right=None):
    claw = claw / np.linalg.norm(claw)
    right = right / np.linalg.norm(right)
    # uses these vectors to construct a rotation matrix -> quaternion.
    # if right is None:
    #     # by default, choose the `up` with the highest possible `z` direction.
    #     if claw[1] < 0:
    #         up_non_ortho = np.array([0, 0, 1])
    #     else:
    #         up_non_ortho = np.array([0, 0, -1])
    #     # subtract the forward component of this vector
    #     proj_onto_forward = np.dot(up_non_ortho, claw) * claw
    #     right = up_non_ortho - proj_onto_forward
    #     right = right / np.linalg.norm(right)

    palm = np.cross(right, claw)
    matrix = np.array([
        [palm[0], right[0], claw[0]],
        [palm[1], right[1], claw[1]],
        [palm[2], right[2], claw[2]],
    ])
    # rotation_matrix = np.array([forward, up, right])
    print(matrix)
    
    return Rotation.from_matrix(matrix).as_quat()
    # return np.array([qw, qx, qy, qz])

def matrix2quat(matrix):
    qx, qy, qz, qw = Rotation.from_matrix(matrix).as_quat()
    return np.array([qw, qx, qy, qz])

def main():
    panda = Panda()
    panda.move_to([0.4, 0.0, 0.2])
    # Base
    # panda.rotate_to(vector2quat(claw=[0, 0, -1], right=[0, -1, 0]))
    panda.rotate_to(vector2quat(claw=[1, 0, 0], right=[0, 1, 0]))
    pos, quat = panda.get_pos()
    # print((Rotation.from_quat(quat.numpy()) * panda.rotation_bias.inv()).as_matrix())
    print("Current matrix [reported]:")
    print((Rotation.from_quat(quat.numpy())).as_matrix())
    print("Current matrix [fixed]:")
    print((Rotation.from_quat(quat.numpy()) * panda.rotation_bias.inv()).as_matrix())
    # print((Rotation.from_quat(quat.numpy())).as_matrix())

"""
In the standard form, we have:
[1,  0,  0]
[0, -1,  0]
[0,  0, -1]
Meaning...
- Gripper x is facing the robot +x direction
- Gripper y is facing the robot -y direction
- Gripper z is facing the robot -z direction

If we have the gripper rotated directly from this until it faces itself, we have:
[0, 0, -1]
[0, -1,  0]
[-1, 0,  0]
Meaning...
- Gripper x is facing the -z direction (downward) -> meaning gripper x is the "forward" 
- Gripper y is facing the -y direction (to the right)
- Gripper z is facing the -x direction (towards the robot)

If we roll the robot so the claws are facing the left, we have:
[1, 0,  0]
[0, 0, -1]
[0, 1,  0]

This information tells us that the rotatoin matrix brings us from gripper frame to robot frame.
Also, strangely, there seems to be a rotation by pi/4 around the z axis... which we need to remove?
Idk why that's there.

Therefore, now let's say we want to construct a rotation matrix based on our knowledge of the gripper's claw
direction (gripper z) and right direction (gripper y).

Then, our rotation matrix should be (to bring from gripper frame to robot frame):
[ .  right_x  claw_x]
[ .  right_y  claw_y]
[ .  right_z  claw_z]

"""

if __name__ == '__main__':
    main()

