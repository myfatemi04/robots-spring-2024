from argparse import ArgumentParser
from time import sleep

from frankx import Affine, Robot


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--host', default='192.168.1.162', help='FCI IP of the robot')
    args = parser.parse_args()

    robot = Robot(args.host)

    while True:
        state = robot.read_once()
        pose = robot.current_pose()
        print("Pose:", (pose.x, pose.y, pose.z))
        # print("Quaternion:", pose.quaternion())
        # print('\nPose: ', robot.current_pose())
        # print('O_TT_E: ', state.O_T_EE)
        # print('Joints: ', state.q)
        # print('Elbow: ', state.elbow)
        sleep(0.05)
