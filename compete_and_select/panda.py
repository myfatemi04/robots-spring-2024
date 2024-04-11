import polymetis
import torch

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
        # Kx_default: [750, 750, 750, 15, 15, 15]
        p = 1500
        p2 = 30
        self.Kx = torch.tensor([p, p, p, p2, p2, p2], dtype=torch.float)

    def get_pos(self):
        pos, rotation = self.robot.get_ee_pose()
        return (pos - self.movement_bias, rotation)

    def move_to(self, pos, **kwargs):
        pos = torch.tensor(pos).float()
        self.robot.move_to_ee_pose(pos + self.movement_bias, Kx=self.Kx, **kwargs)
        
    def move_by(self, pos, **kwargs):
        pos = torch.tensor(pos).float()
        self.robot.move_to_ee_pose(pos, delta=True, Kx=self.Kx, **kwargs)

    def rotate_to(self, quat, **kwargs):
        quat = torch.tensor(quat).float()
        pos, _ = self.robot.get_ee_pose()
        self.robot.move_to_ee_pose(pos, quat, **kwargs)

    def rotate_by(self, quat):
        quat = torch.tensor(quat).float()
        zero = torch.tensor([0, 0, 0]).float()
        self.robot.move_to_ee_pose(zero, quat, delta=True)

    def start_grasp(self):
        self.gripper.grasp(speed=1, force=1, grasp_width=0.00)

    def stop_grasp(self):
        self.gripper.grasp(speed=1, force=1, grasp_width=0.08)
