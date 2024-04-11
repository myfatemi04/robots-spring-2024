import time
import polymetis
import torch

polymetis_server_ip = "192.168.1.222"
robot = polymetis.RobotInterface(
    ip_address=polymetis_server_ip,
    port=50051,
    enforce_version=False,
)

robot.move_to_ee_pose(torch.tensor([0.5, 0.0, 0.5]).float())

gripper = polymetis.GripperInterface(
    ip_address=polymetis_server_ip,
    port=50052,
)

ROBOT_CONTROL_X_BIAS = 0.14
ROBOT_CONTROL_Y_BIAS = 0.03
ROBOT_CONTROL_Z_BIAS = 0.10
movement_bias = torch.tensor([ROBOT_CONTROL_X_BIAS, ROBOT_CONTROL_Y_BIAS, ROBOT_CONTROL_Z_BIAS]).float()

for _ in range(4):
    print("close")
    gripper.grasp(2, 1.0, 0)
    time.sleep(3)
    print("open")
    gripper.grasp(2, 1.0, 0.08)
    time.sleep(3)

