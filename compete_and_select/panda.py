import math
import polymetis
import torch
import numpy as np
from scipy.spatial.transform import Rotation

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

        ROBOT_CONTROL_X_BIAS = 0.17
        ROBOT_CONTROL_Y_BIAS = 0.025
        ROBOT_CONTROL_Z_BIAS = 0.10
        self.movement_bias = torch.tensor([ROBOT_CONTROL_X_BIAS, ROBOT_CONTROL_Y_BIAS, ROBOT_CONTROL_Z_BIAS]).float()
        # self.rotation_bias = Rotation.from_quat(np.array([0, 0, math.sin(math.pi/2), math.cos(math.pi/2)]))
        self.rotation_bias = Rotation.from_matrix(np.array([
            [np.sqrt(2)/2, np.sqrt(2)/2, 0],
            [-np.sqrt(2)/2, np.sqrt(2)/2, 0],
            [0, 0, 1]
        ]))
        # Kx_default: [750, 750, 750, 15, 15, 15]
        p = 1500
        p2 = 30
        self.Kx = torch.tensor([p, p, p, p2, p2, p2], dtype=torch.float)
        self.time_to_go = 8

    def get_pos(self):
        pos, rotation = self.robot.get_ee_pose()
        return (pos - self.movement_bias, rotation)

    def move_to(self, pos, **kwargs):
        pos = torch.tensor(pos).float()
        self.robot.move_to_ee_pose(pos + self.movement_bias, time_to_go=self.time_to_go, Kx=self.Kx, **kwargs)
        
    def move_by(self, pos, **kwargs):
        pos = torch.tensor(pos).float()
        self.robot.move_to_ee_pose(pos, delta=True, time_to_go=self.time_to_go, Kx=self.Kx, **kwargs)

    def rotate_to(self, quat, **kwargs):
        quat = torch.tensor(quat).float()
        pos, _ = self.robot.get_ee_pose()

        orig_quat = quat

        # apply rotation correction
        quat = torch.tensor((self.rotation_bias * Rotation.from_quat(quat.detach().cpu().numpy())).as_quat(), dtype=torch.float)

        print(quat, orig_quat)

        self.robot.move_to_ee_pose(pos, quat, time_to_go=self.time_to_go, **kwargs)

    def rotate_by(self, quat):
        quat = torch.tensor(quat).float()
        zero = torch.tensor([0, 0, 0]).float()
        self.robot.move_to_ee_pose(zero, quat, time_to_go=self.time_to_go, delta=True)

    def start_grasp(self):
        self.gripper.grasp(speed=1, force=1, grasp_width=0.00)

    def stop_grasp(self):
        self.gripper.grasp(speed=1, force=1, grasp_width=0.08)
