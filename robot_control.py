from polymetis import RobotInterface

robot = RobotInterface(
    ip_address="localhost",
)
"""
Available in RobotInterface:
https://facebookresearch.github.io/fairo/polymetis/polymetis.html#polymetis.robot_interface.RobotInterface
 - get_robot_state() -> RobotState
 - get_ee_pose() -> [position[3], quaternion[4]]
 - move_to_ee_pose(position, orientation, time_to_go) -> RobotState[]

RobotState is specified with Protobuf.
The proto file is here: https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/protos/polymetis.proto
message RobotState {
  // Contains the robot state. Fields are optional
  // depending on context. Add to this message to
  // extend for different types of robots/sensors.
  google.protobuf.Timestamp timestamp = 1;
  repeated float joint_positions = 2;
  repeated float joint_velocities = 3;
  repeated float joint_torques_computed = 4;              //Torques received from the controller server (copy of TorqueCommand.joint_torques)
  repeated float prev_joint_torques_computed = 5;         //Torques received from the controller server in the previous timestep
  repeated float prev_joint_torques_computed_safened = 6; //Torques after adding safety mechanisms in the previous timestep
  repeated float motor_torques_measured = 7;              //Measured torque signals from the robot motors
  repeated float motor_torques_external = 8;              //Measured external torques exerted on the robot motors
  repeated float motor_torques_desired = 9;               //Desired torques signals from the robot motors
  float prev_controller_latency_ms = 10;                  //Latency of previous ControlUpdate call
  bool prev_command_successful = 11;                      //Whether previous command packet is successfully transmitted
  int32 error_code = 12;
}

There is surely a way to calculate XYZ position from joint positions; just a couple matrix multiplications.
Working backwards from the end-effector, each joint position translates the end-effector (relatively) and then rotates along
the joint's rotation axis. Then, the joint before that applies *it's* translation and rotation, ...

"""

### For controlling gripper
from polymetis import GripperInterface

polymetis_server_ip = "192.168.1.222"

gripper = GripperInterface(
    ip_address=polymetis_server_ip,
)

# example usages
gripper_state = gripper.get_state()
gripper.goto(width=0.01, speed=0.05)
gripper.grasp(speed=0.05, force=0.1)
