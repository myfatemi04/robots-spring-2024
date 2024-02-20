import polymetis
import torch

polymetis_server_ip = "192.168.1.222"
robot = polymetis.RobotInterface(
  ip_address=polymetis_server_ip,
  port=50051,
  enforce_version=False,
)
ROBOT_CONTROL_X_BIAS = 0.14
ROBOT_CONTROL_Y_BIAS = 0.03
ROBOT_CONTROL_Z_BIAS = 0.10

def move(x, y, z):
    robot.move_to_ee_pose(torch.tensor([x + ROBOT_CONTROL_X_BIAS, y + ROBOT_CONTROL_Y_BIAS, z + ROBOT_CONTROL_Z_BIAS]).float())

move(0.4, 0.0, 0.4)