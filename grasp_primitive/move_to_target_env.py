import time

import gymnasium as gym
import torch
from gymnasium import spaces


class PandaEnv(gym.Env):
    def __init__(self, robot, cam, render_mode=None):
        from polymetis import RobotInterface

        # the more actions we add, the more learning needs to occur.
        # nop, forward, backward, left, right
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(640, 480, 3))
        self.render_mode = render_mode
        self.robot: RobotInterface = robot
        self.robot.go_home()

        control_kp = torch.tensor([80, 80, 80, 50.0, 40.0, 50.0]) / 2
        control_kv = torch.ones((6,)) * torch.sqrt(control_kp)

        self.robot.start_cartesian_impedance(control_kp, control_kv)
        self.cam = cam
        self.internal_ee_pose, self.internal_ee_quat = self.robot.get_ee_pose()

        self.min_pose = torch.tensor([0.1, -0.5, 0.2])
        self.max_pose = torch.tensor([0.7, 0.5, 0.5])

    def step(self, action):
        distance = 0.025

        prev_pos = self.internal_ee_pose.clone()

        if action == 0:
            # do nothing
            pass
        elif action == 1:
            # left
            self.internal_ee_pose[1] -= distance
        elif action == 2:
            # right
            self.internal_ee_pose[1] += distance
        elif action == 3:
            # forward
            self.internal_ee_pose[0] += distance
        elif action == 4:
            # backward
            self.internal_ee_pose[0] -= distance

        print(self.internal_ee_pose)

        self.internal_ee_pose = torch.clamp(self.internal_ee_pose, self.min_pose, self.max_pose)

        # do a sort of smooth move
        delta = self.internal_ee_pose - prev_pos

        if torch.any(delta != 0):
            for i in range(11):
                interpolated_pose = prev_pos + delta * i / 10
                self.robot.update_desired_ee_pose(
                    interpolated_pose,
                    self.internal_ee_quat
                )
                time.sleep(0.05)
        
        images = next(self.cam)

        return images, 0, False, {}

# test as a human
def test_as_human():
    from compete_and_select.perception.realsense import RealsenseWrapper
    from polymetis import RobotInterface

    cam = RealsenseWrapper()
    robot = RobotInterface(ip_address="192.168.1.222", port=50051, enforce_version=False)
    env = PandaEnv(robot, cam)

    for i in range(100):
        action = int(input("Type an action [0, 1, 2, 3, 4]:"))

        (color, depth), reward, done, info = env.step(action)
        print("reward", reward)
        print("done", done)
        print("info", info)

        # import matplotlib.pyplot as plt

        # plt.imshow(color)
        # plt.show()

        time.sleep(0.5)

if __name__ == '__main__':
    test_as_human()
