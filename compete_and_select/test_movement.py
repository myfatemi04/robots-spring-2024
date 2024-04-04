# import frankx

# robot = frankx.Robot('192.168.1.162')
# robot.set_dynamic_rel(0.05)
# robot.move(frankx.LinearMotion(frankx.Affine(0.5, 0.5, 0.5)))

# input("[press enter] ")

### We use Polymetis instead. Should create some bridge to shield from differences in robot inferfaces.
import polymetis
import torch

# Update this according to your own setup.
polymetis_server_ip = "192.168.1.222"
robot = polymetis.RobotInterface(
  ip_address=polymetis_server_ip,
  port=50051,
  enforce_version=False,
)

ROBOT_CONTROL_X_BIAS = 0#.14
ROBOT_CONTROL_Y_BIAS = 0#.03
ROBOT_CONTROL_Z_BIAS = 0#.10

def move(x, y, z):
    robot.move_to_ee_pose(torch.tensor([x + ROBOT_CONTROL_X_BIAS, y + ROBOT_CONTROL_Y_BIAS, z + ROBOT_CONTROL_Z_BIAS]).float())

move(0.5, 0.0, 0.5)
