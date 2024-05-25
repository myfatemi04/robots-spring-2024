# We might get images and actions at different times.
# One thing we can do to make things faster is remove no-op actions,
# because we know the environment will stay static.

import time

from polymetis import GripperInterface, RobotInterface

from compete_and_select.teleoperation.robot_config import RobotConfig
from compete_and_select.teleoperation.teleoperation import (
    JoystickController, TeleoperationInterface)
from compete_and_select.perception.realsense import show_realsense_stream

# Initialize robot interface
server_ip = "192.168.1.222"
gripper = GripperInterface(ip_address=server_ip, port=50052)
robot = RobotInterface(ip_address=server_ip, enforce_version=False, port=50051)
cfg = RobotConfig()
device = JoystickController()
teleoperation_interface = TeleoperationInterface(robot, gripper, device, hz=50)

# Reset
robot.go_home()
time.sleep(2.0)

robot.start_cartesian_impedance(Kx=cfg.control_kp, Kxd=cfg.control_kv)

teleoperation_interface.start()

show_realsense_stream()

teleoperation_interface.stop()

