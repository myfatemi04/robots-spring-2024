import math
import polymetis
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import time

class Panda:
    def __init__(self, polymetis_server_ip="192.168.1.222"):
        self.robot = polymetis.RobotInterface(
            ip_address=polymetis_server_ip,
            port=50051,
            enforce_version=False,
        )
        self.gripper = polymetis.GripperInterface(
            ip_address=polymetis_server_ip,
            port=50052,
        )

        ROBOT_CONTROL_X_BIAS = 0.18
        ROBOT_CONTROL_Y_BIAS = 0.0
        ROBOT_CONTROL_Z_BIAS = 0.10
        self.movement_bias = torch.tensor([ROBOT_CONTROL_X_BIAS, ROBOT_CONTROL_Y_BIAS, ROBOT_CONTROL_Z_BIAS]).float()
        # self.rotation_bias = Rotation.from_quat(np.array([0, 0, math.sin(math.pi/2), math.cos(math.pi/2)]))
        self.rotation_bias = Rotation.from_matrix(np.array([
            # [1, 0, 0],
            # [0, 1, 0],
            # [0, 0, 1],
            [np.sqrt(2)/2, -np.sqrt(2)/2, 0],
            [np.sqrt(2)/2, np.sqrt(2)/2, 0],
            [0, 0, 1]
        ]))
        self.robot.Kq_default = torch.tensor([40, 30, 50, 25, 35, 25, 10], dtype=torch.float)
        self.robot.Kqd_default = torch.tensor([4, 6, 5, 5, 3, 2, 1], dtype=torch.float)
        self.robot.Kx_default = torch.tensor([650, 650, 650, 50, 50, 50], dtype=torch.float)
        self.robot.Kxd_default = torch.tensor([40, 40, 40, 10, 10, 10], dtype=torch.float)
        self.time_to_go = 8

    def get_pos(self):
        pos, rotation = self.robot.get_ee_pose()
        return (pos - self.movement_bias, rotation)

    def move_to(self, pos, orientation=None, **kwargs):
        pos = torch.tensor(pos).float()

        if orientation is not None:
            # fix the weird rotation bug
            quat = orientation
            quat = torch.tensor(quat).float()
            orientation = (Rotation.from_quat(quat.detach().cpu().numpy()) * self.rotation_bias).as_quat()

        self.robot.move_to_ee_pose(pos + self.movement_bias, orientation, time_to_go=self.time_to_go, **kwargs)
        
    def move_by(self, pos, **kwargs):
        pos = torch.tensor(pos).float()
        self.robot.move_to_ee_pose(pos, delta=True, time_to_go=self.time_to_go, **kwargs)

    def rotate_to(self, quat, **kwargs):
        quat = torch.tensor(quat).float()
        pos, _ = self.robot.get_ee_pose()

        target_rot = Rotation.from_quat(quat.detach().cpu().numpy()) * self.rotation_bias

        print("Target mat:")
        print(target_rot.as_matrix())

        # apply rotation correction
        quat = torch.tensor(target_rot.as_quat(), dtype=torch.float)

        self.robot.move_to_ee_pose(pos, quat, time_to_go=self.time_to_go, **kwargs)

    def rotate_by(self, quat):
        quat = torch.tensor(quat).float()
        zero = torch.tensor([0, 0, 0]).float()
        self.robot.move_to_ee_pose(zero, quat, time_to_go=self.time_to_go, delta=True)

    def start_grasp(self):
        # speed, force, grasp width
        self.gripper.grasp(2, 1.0, 0)
        time.sleep(0.5)

    def stop_grasp(self):
        self.gripper.grasp(2, 1.0, 0.08)
        time.sleep(0.5)
