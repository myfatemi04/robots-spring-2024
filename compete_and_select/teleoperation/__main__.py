import os
import pickle
import time
from matplotlib import pyplot as plt

from polymetis import GripperInterface, RobotInterface

from ..perception.rgbd import RGBD
from .joystick_controller import JoystickController
from .robot_config import RobotConfig
from .teleoperation import TeleoperationInterface


def main():

    # Initialize robot interface
    server_ip = "192.168.1.222"
    gripper = GripperInterface(ip_address=server_ip, port=50052)
    robot = RobotInterface(ip_address=server_ip, enforce_version=False, port=50051)
    cfg = RobotConfig()
    device = JoystickController()
    teleoperation_interface = TeleoperationInterface(robot, gripper, device, hz=50)

    rgbd = RGBD(num_cameras=1, auto_calibrate=False)
    with open("calibrations.pkl", "rb") as f:
        calibrations = pickle.load(f)
        for i, calibration in enumerate(calibrations[:len(rgbd.cameras)]):
            rgbd.cameras[i].extrinsic_matrix = calibration

    print("Restored calibrations.")

    # Reset
    robot.go_home()
    time.sleep(2.0)
    
    robot.start_cartesian_impedance(Kx=cfg.control_kp, Kxd=cfg.control_kv)

    teleoperation_interface.start()

    i = 0
    while os.path.exists(f"teleoperations/{i}"):
        i += 1
    
    out_dir = f"teleoperations/{i}"
    os.makedirs(out_dir)

    event_counter = 0

    cmd = None
    try:
        while cmd != "exit":
            cmd = input("> ")
            event_counter += 1

            if cmd == 'c':
                # Capture an RGBD image and save it.
                rgbs, pcds = rgbd.capture()
                with open(os.path.join(out_dir, f"{event_counter:03d}_rgbd.pkl"), "wb") as f:
                    pickle.dump((rgbs, pcds), f)

                plt.title("Captured Image")
                plt.imshow(rgbs[0])
                plt.show()

            elif cmd == 'g':
                # Capture the current robot state.
                with open(os.path.join(out_dir, f"{event_counter:03d}_state.pkl"), "wb") as f:
                    pos, quat = robot.get_ee_pose()
                    pickle.dump({"pos": pos.tolist(), "quat_wxyz": quat.tolist()}, f)

                    print("Captured position:", pos.tolist())
                    print("Captured quaternion:", quat.tolist())
    finally:
        teleoperation_interface.stop()

if __name__ == "__main__":
    main()
