import frankx
import numpy as np
from scipy.spatial.transform import Rotation
import time

rb = frankx.Robot("192.168.1.163")
state = rb.get_state()
print(dir(state))
print(np.array(state.O_T_EE).reshape(4, 4).T)
print(np.array(state.O_T_EE_c).reshape(4, 4).T)
print(np.array(state.O_T_EE_d).reshape(4, 4).T)
print(np.array(state.F_T_EE).reshape(4, 4).T)
# print(np.array(state.F_T_NE).reshape(4, 4).T)


ang = np.pi

target_matrix = np.array([
    [1, 0, 0],
    [0, np.cos(ang), -np.sin(ang)],
    [0, np.sin(ang), np.cos(ang)],
])

# target_matrix = np.array([
#     [0, 1, 0],
#     [1, 0, 0],
#     [0, 0, -1],
# ])

print(target_matrix)

rot = Rotation.from_matrix(target_matrix)
qx, qy, qz, qw = rot.as_quat()

# print(qx, qy, qz, qw)

# for _ in range(3):
rb.recover_from_errors()
rb.set_dynamic_rel(0.1)
rb.move(frankx.LinearMotion(frankx.Affine(0.4, 0.0, 0.4, qw, qx, qy, qz)))

# input()

exit()

gp = frankx.Gripper('192.168.1.163')
gp.move(0.08)
print(gp.max_width)
print(gp.has_error)
print(gp.width())

exit()

# Create a rotation matrix that "looks at" a certain direction

ang = np.pi

target_matrix = np.array([
    [1, 0, 0],
    [0, np.cos(ang), -np.sin(ang)],
    [0, np.sin(ang), np.cos(ang)],
])

print(target_matrix)

rot = Rotation.from_matrix(target_matrix)
qx, qy, qz, qw = rot.as_quat()

robot = frankx.Robot('192.168.1.162')
robot.recover_from_errors()
robot.set_dynamic_rel(0.05)

state = robot.get_state()
print(np.array(state.O_T_EE).reshape(4, 4).T) # column-major format

# This motion occurs in end-effector frame.
# robot.move(frankx.LinearRelativeMotion(frankx.Affine(0, -0.05, 0)))
robot.move(frankx.LinearMotion(frankx.Affine(0.6, 0.0, 0.4, qw, qx, qy, qz)))


# gripper = frankx.Gripper('192.168.1.162')
# gripper.clamp()

# time.sleep(1)

# gripper.release()
