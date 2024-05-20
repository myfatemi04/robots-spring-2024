import frankx
import numpy as np
from scipy.spatial.transform import Rotation
import time

gp = frankx.Gripper('192.168.1.162')
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

