"""
This code is adapted from FurnitureBench: https://github.com/clvrai/furniture-bench
"""

import time

import numpy as np
import torch
from polymetis import GripperInterface, RobotInterface
import torchcontrol as toco

from .joystick_controller import JoystickController
from .operational_space_controller import init_controller
from .robot_config import RobotConfig


def quat_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float32,
    )

# Can be jitted.
# @numba.jit(nopython=True, cache=True)
def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.
    Args:
        rmat: 3x3 rotation matrix
    Returns:
        vec4 float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]

def euler2mat(euler):
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat

if __name__ == "__main__":
    # Initialize robot interface
    server_ip = "192.168.1.222"
    gripper = GripperInterface(ip_address=server_ip, port=50052)
    robot = RobotInterface(ip_address=server_ip, enforce_version=False, port=50051)

    # Reset
    robot.go_home()
    time.sleep(2.0)

    POSITION_CONTROL = 30 * 5
    ROTATION_CONTROL = 15 * 5
    GRASP_CLOSED = gripper.get_state().width < 0.075

    cfg = RobotConfig()
    device = JoystickController()
    # init_controller(robot, cfg)
    # robot.send_torch_policy(toco.policies.CartesianImpedanceControl(cfg.control_kp, cfg.control_kv))
    robot.start_cartesian_impedance(Kx=cfg.control_kp, Kxd=cfg.control_kv)

    # Smoohten ee_pos over time.
    ee_pos, ee_quat = robot.get_ee_pose()
    beta = 0.2
    grip_force = 2.0

    hz_des = 50

    try:
        while True:
            loop_start = time.time()

            new_ee_pos, new_ee_quat = robot.get_ee_pose()

            device_data, robot_action = device.get_data()
            # Filtering
            
            arm_action, grasp = robot_action[:-1], robot_action[-1]

            # x,y, z movement scaling
            arm_pose = arm_action[:3] / POSITION_CONTROL

            # roll, pitch, yaw movement scaling
            act_rot = arm_action[3:] / ROTATION_CONTROL

            # Prevent drift, and only update the desired dimensions of the EE pos.
            # ee_pos[arm_pose != 0] = beta * ee_pos[arm_pose != 0] + (1 - beta) * new_ee_pos[arm_pose != 0]
            # if np.any(act_rot != 0):
            #     ee_quat = beta * ee_quat + (1 - beta) * new_ee_quat

            # roll, pitch, yaw to quarternion conversion
            act_quat = mat2quat(euler2mat(act_rot))
            
            # goal xyz pose and quaternion calculation
            goal_ee_pos = torch.tensor(ee_pos, dtype=torch.float32) + torch.tensor(arm_pose, dtype=torch.float32)
            goal_ee_quat = torch.tensor(quat_multiply(ee_quat, act_quat), dtype=torch.float32)

            if (goal_ee_pos != ee_pos).any():
                ee_pos = goal_ee_pos
            if (goal_ee_quat != ee_quat).any():
                ee_quat = goal_ee_quat

            # gripper open-close execute
            if grasp:
                if grasp == -1 and GRASP_CLOSED == True:
                    gripper.grasp(speed=5, force=grip_force, grasp_width=0.08, blocking=False)
                    print("Opening grasp.")
                    time.sleep(2)
                    GRASP_CLOSED = False

                elif grasp == 1 and GRASP_CLOSED == False:
                    gripper.grasp(speed=5, force=grip_force, grasp_width=0.0, blocking=False)
                    print("Closing grasp.")
                    time.sleep(2)
                    GRASP_CLOSED = True
                else:
                    pass

            # update xyz pose and quaternion
            # robot.move_to_ee_pose(position=goal_ee_pos, orientation=goal_ee_quat)
            robot.update_desired_ee_pose(position=goal_ee_pos, orientation=goal_ee_quat)

            loop_end = time.time()
            expected_delay = 1 / hz_des
            # print(f"Current pose: {goal_ee_pos}, Loop time: {loop_end - loop_start:.3f}")

            time.sleep(max(0, expected_delay - (loop_end - loop_start)))
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        robot.terminate_current_policy()
