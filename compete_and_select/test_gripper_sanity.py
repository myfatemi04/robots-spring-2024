import time
import polymetis
import torch

polymetis_server_ip = "192.168.1.222"
# robot = polymetis.RobotInterface(
#     ip_address=polymetis_server_ip,
#     port=50051,
#     enforce_version=False,
# )
gripper = polymetis.GripperInterface(
    ip_address=polymetis_server_ip,
    port=50052,
)

ROBOT_CONTROL_X_BIAS = 0.14
ROBOT_CONTROL_Y_BIAS = 0.03
ROBOT_CONTROL_Z_BIAS = 0.10
movement_bias = torch.tensor([ROBOT_CONTROL_X_BIAS, ROBOT_CONTROL_Y_BIAS, ROBOT_CONTROL_Z_BIAS]).float()

for _ in range(3):
    print("go")
    gripper.grasp(5, 1.0, 0, blocking=False)
    time.sleep(2)
    print(gripper.get_state())
    gripper.grasp(5, 1.0, 0.08, blocking=False)
    time.sleep(4)

