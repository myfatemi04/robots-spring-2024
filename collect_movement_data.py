import json
import os
import sys
from time import sleep

import cv2
import numpy as np
import time

from frankx import Affine, Robot
from compete_and_select.perception.camera import get_cameras

def collect_trajectories():
    robot = Robot('192.168.1.162')
    left, right = get_cameras()

    run = sys.argv[1]

    prefix = f'./labels/control/{run}'

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    out_left = cv2.VideoWriter(os.path.join(prefix, 'output_left.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1280, 720))
    out_right = cv2.VideoWriter(os.path.join(prefix, 'output_right.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1280, 720))

    times = []
    poses = []
    quaternions = []
    gripper_widths = []

    GRIPPER_FULLY_CLOSED_WIDTH = -0.020493666365742683
    GRIPPER_FULLY_OPEN_WIDTH = 0.059208593606948856

    while True:
        # https://github.com/pantor/frankx/tree/master
        state = robot.read_once()
        # Type: https://github.com/pantor/affx/blob/main/src/python.cpp
        ee_state: Affine = robot.current_pose()

        # Record camera states
        left_color = np.ascontiguousarray(left.capture().color[:, :, :3])
        right_color = np.ascontiguousarray(right.capture().color[:, :, :3])

        out_left.write(left_color)
        out_right.write(right_color)

        """
        Also available:
        * O_T_EE [state.O_T_EE]
        * Joints [state.q]
        * Elbow [state.elbow]
        """
        pose = (ee_state.x, ee_state.y, ee_state.z)
        quaternion = ee_state.quaternion()
        gripper_width = robot.get_gripper().width()
        
        times.append(time.time())
        poses.append(pose)
        quaternions.append(quaternion)
        gripper_widths.append(gripper_width)

        cv2.imshow('Left', left_color)
        cv2.imshow('Right', right_color)
        if cv2.waitKey(1) == ord('q'):
            break

        sleep(0.05)

    out_left.release()
    out_right.release()

    with open(os.path.join(prefix, "data.json"), "w") as f:
        json.dump({
            "times": times,
            "poses": poses,
            "quaternions": quaternions,
            "gripper_widths": gripper_widths,
            "gripper_fully_closed_width": GRIPPER_FULLY_CLOSED_WIDTH,
            "gripper_fully_open_width": GRIPPER_FULLY_OPEN_WIDTH,
        }, f)

def annotate_instructions_and_points():
    point = None

    def select_point(event, x, y, flags, param):
        nonlocal point

        if event == cv2.EVENT_LBUTTONUP:
            point = (x, y)

    previous_instruction = None

    cv2.namedWindow("Labeling")
    cv2.setMouseCallback("Labeling", select_point)

    if not os.path.exists("labels/control_instructions"):
        os.makedirs("labels/control_instructions")

    for demonstration in os.listdir("labels/control"):
        # Show image and allow user to interact with window

        cap = cv2.VideoCapture(f"labels/control/{demonstration}/output_left.mp4")
        # Skip first 2 frames to allow camera to adjust
        cap.read()
        cap.read()
        ret, left_first_frame = cap.read()
        cap.release()

        cap = cv2.VideoCapture(f"labels/control/{demonstration}/output_right.mp4")
        # Skip first 2 frames to allow camera to adjust
        cap.read()
        cap.read()
        ret, right_first_frame = cap.read()
        cap.release()

        cv2.imwrite(f"labels/control_instructions/{demonstration}_start_position_left.png", left_first_frame)
        cv2.imwrite(f"labels/control_instructions/{demonstration}_start_position_right.png", right_first_frame)

        if os.path.exists(f"labels/control_instructions/{demonstration}.json"):
            continue

        cv2.imshow("Labeling", left_first_frame)
        cv2.waitKey(0)

        left_point = point
        point = None

        cv2.imshow("Labeling", right_first_frame)
        cv2.waitKey(0)

        right_point = point
        point = None

        instruction = input("Instruction:") or previous_instruction

        annotation = {
            'left_point': left_point,
            'right_point': right_point,
            'instruction': instruction,
        }

        with open(f"labels/control_instructions/{demonstration}.json", "w") as f:
            json.dump(annotation, f)

        previous_instruction = instruction

annotate_instructions_and_points()
