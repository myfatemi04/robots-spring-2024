import json
import os
import pickle
import time

import cv2
import gymnasium as gym
import torch
from gymnasium import spaces
from polymetis import GripperInterface, RobotInterface
from compete_and_select.perception.realsense import RealsenseWrapper


class PandaEnv(gym.Env):
    def __init__(self, robot, gripper, cam, render_mode=None):

        # the more actions we add, the more learning needs to occur.
        # nop, forward, backward, left, right, up, down
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(640, 480, 3))
        self.render_mode = render_mode
        self.robot: RobotInterface = robot
        self.robot.go_home()
        self.gripper: GripperInterface = gripper
        self.last_gripper_ts = 0

        control_kp = torch.tensor([80, 80, 80, 50.0, 40.0, 50.0]) / 2
        control_kv = torch.ones((6,)) * torch.sqrt(control_kp)

        self.robot.start_cartesian_impedance(control_kp, control_kv)
        self.cam: RealsenseWrapper = cam
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
        elif action == 5:
            # up
            self.internal_ee_pose[2] += distance
        elif action == 6:
            # down
            self.internal_ee_pose[2] -= distance
        elif action == 7:
            # gripper.
            # cancel gripper actions if they are too close to each other
            if time.time() - self.last_gripper_ts <= 1:
                action = 0
            else:
                if self.gripper.get_state().width < 0.04:
                    self.gripper.grasp(speed=5, force=2, grasp_width=0.08)
                else:
                    self.gripper.grasp(speed=5, force=2, grasp_width=0)

                self.last_gripper_ts = time.time()

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
    from polymetis import GripperInterface, RobotInterface
    from pynput import keyboard  # type: ignore

    cam = RealsenseWrapper()
    robot = RobotInterface(ip_address="192.168.1.222", port=50051, enforce_version=False)
    gripper = GripperInterface(ip_address="192.168.1.222", port=50052)
    env = PandaEnv(robot, gripper, cam)

    expected_hz = 4

    keys = {}

    def on_press(key):
        keys[key] = True
        
    def on_release(key):
        keys[key] = False

    color_images = []
    depth_images = []
    actions = []

    with keyboard.Listener(on_press, on_release) as listener:
        for i in range(1000):
            start_time = time.time()

            action = 0
            for (i, k) in enumerate([keyboard.Key.left, keyboard.Key.right, keyboard.Key.up, keyboard.Key.down, keyboard.Key.space, keyboard.Key.shift, keyboard.Key.enter]):
                if keys.get(k, False):
                    action = i + 1
                    break

            actions.append(action)

            (color, depth), reward, done, info = env.step(action)

            color_images.append(color)
            depth_images.append(depth)

            end_time = time.time()
            time.sleep(max(0, 1 / expected_hz - (end_time - start_time)))

            # we end the episode for the "enter" action.
            if action == 7:
                break

    env.cam.running = False
    env.cam.thread.join()

    # remove the current grasp
    gripper.grasp(speed=5, force=2, grasp_width=0.08)

    # save the trajectory
    i = 0
    while os.path.exists(f"trajectories/{i}"):
        i += 1
    
    print("Saving to trajectories/{}".format(i))
    # if 'y' != input('OK? (y/n) '):
    #     return

    os.makedirs(f"trajectories/{i}", exist_ok=True)

    with open(f"trajectories/{i}/actions.json", "w") as f:
        json.dump(actions, f)

    for j, (color, depth) in enumerate(zip(color_images, depth_images)):
        cv2.imwrite(f"trajectories/{i}/{j:03d}_color.png", color)
        path = f"trajectories/{i}/{j:03d}_depth.pkl"
        with open(path, "wb") as f:
            pickle.dump(depth, f)

if __name__ == '__main__':
    test_as_human()
