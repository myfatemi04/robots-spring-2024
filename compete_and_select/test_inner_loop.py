# Given a goal position and instruction, we try to adjust the robot's end effector position to match.
# Maybe we don't even need to specify goals in exact 3D coordinates.

import os
import pickle
import sys

import numpy as np
import PIL.Image as Image
import polymetis
from matplotlib import pyplot as plt
import torch
from vlms import gpt4v_plusplus as gpt4v

task_image = Image.open("sample_images/task_image.png")

import pyk4a

sys.path.insert(0, "../")
from camera import Camera

num_cameras = 1
if pyk4a.connected_device_count() < num_cameras:
    print(f"Error: Not enough K4A devices connected (<{num_cameras}).")
    exit(1)

k4a_devices = [pyk4a.PyK4A(device_id=i) for i in range(num_cameras)]
k4a_device_map = {}
for device in k4a_devices:
    device.start()
    k4a_device_map[device.serial] = device

# left = Camera(k4a_device_map['000256121012'])
# right = Camera(k4a_device_map['000243521012'])
camera = Camera(k4a_devices[0])
camera.import_calibration(pickle.load(open("calib.pkl", "rb")))

polymetis_server_ip = "192.168.1.222"
robot = polymetis.RobotInterface(
    ip_address=polymetis_server_ip,
    port=50051,
    enforce_version=False,
)
ROBOT_CONTROL_X_BIAS = 0.14
ROBOT_CONTROL_Y_BIAS = 0.03
ROBOT_CONTROL_Z_BIAS = 0.10
movement_bias = torch.tensor([ROBOT_CONTROL_X_BIAS, ROBOT_CONTROL_Y_BIAS, ROBOT_CONTROL_Z_BIAS]).float()

rot_matrix = camera.extrinsic_matrix[:3, :3]

camera_directions = {
    'forward': rot_matrix[2],
    'backward': -rot_matrix[2],
    'left': -rot_matrix[0],
    'right': rot_matrix[0],
    'up': -rot_matrix[1],
    'down': rot_matrix[1],
}
robot_directions = {
    'forward': np.array([1, 0, 0]),
    'backward': np.array([-1, 0, 0]),
    'left': np.array([0, 1, 0]),
    'right': np.array([0, -1, 0]),
    'up': np.array([0, 0, 1]),
    'down': np.array([0, 0, -1]),
}

def calculate_translation_vector(direction, amount_m):
    direction_vector = robot_directions[direction]
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    move = direction_vector * amount_m
    return move

def move_relative(move):
    move_tens = torch.tensor([move[0], move[1], move[2]], dtype=torch.float)
    print("Moving by", move)
    if 'y' != input("Final confirm (y/n) "):
        return
    robot.move_to_ee_pose(move_tens, delta=True)

move_arm_function = {
    "type": "function",
    "function": {
        "description": "Adjusts the robot arm's position.",
        "name": "move_arm",
        "parameters": {
            "type": "object",
            "properties": {
                "thought_process": {"type": "string"},
                "movement_direction": {"type": "string"},
                "movement_amount_centimeters": {"type": "number"}
            },
            "required": ["thought_process", "movement_direction", "movement_amount_centimeters"]
        }
    }
}

def render_virtual_plan(image, plan):
    # Add a small correction step.
    current_eef_pose, current_eef_quat = robot.get_ee_pose()
    current_eef_pose = current_eef_pose - movement_bias
    translation_vector = calculate_translation_vector(plan['movement_direction'], plan['movement_amount_centimeters'] / 100)

    arrow_start_pt = camera.project_points(current_eef_pose.unsqueeze(0).numpy())[0]
    arrow_end_pt = camera.project_points(current_eef_pose.unsqueeze(0).numpy() + translation_vector)[0]

    # draw plan
    scale = 1
    plt.imshow(image)
    plt.arrow(arrow_start_pt[0], arrow_start_pt[1], scale * (arrow_end_pt[0] - arrow_start_pt[0]), scale * (arrow_end_pt[1] - arrow_start_pt[1]), label='planned movement', width=5, head_width=20, edgecolor='red', facecolor='red')
    plt.savefig("tmp.png")
    virtual_plan_image = Image.open("tmp.png")
    os.remove("tmp.png")
    plt.show()

    return (virtual_plan_image)

def prompt_with_virtual_plans(task_image, previous_plans, current_image):
    # use in-context learning by rendering the plan virtually
    # sort of a notion of visual prompt engineering / providing a correction step
    prompt_messages = [
        ("Look at this image. You are tasked with grabbing the selected item.", task_image),
        None,
        ("Here is the current state of the scene. Where is the target relative to the robot's gripper? To achieve the target, should the arm move up, down, left, right, forward, or backward? Answer with a single direction and a number of centimeters.",
        current_image)
    ]
    for previous_plan in previous_plans:
        prompt_messages.append(f"Movement direction: {previous_plan['movement_direction']}, Movement amount: {previous_plan['movement_amount_centimeters']}")
        prompt_messages.append((render_virtual_plan(image, previous_plan), f"This red arrow is what the \"{previous_plan['movement_direction']}\" direction looks like, and its length is {previous_plan['movement_amount_centimeters']}. Given this knowledge, choose the movement direction and magnitude you think is best. You can choose the same movement direction and magnitude if you want."))
    return prompt_messages

image_np = np.ascontiguousarray(camera.capture().color[..., :-1][..., ::-1])
image = Image.fromarray(image_np)

plt.subplot(1, 2, 1)
plt.title("Task image")
plt.imshow(task_image)
plt.subplot(1, 2, 2)
plt.title("Current Image")
plt.imshow(image)
plt.show()

previous_plans = []

for _ in range(3):
    plan = gpt4v(prompt_with_virtual_plans(task_image, previous_plans, image),
        tools=[move_arm_function],
        tool_choice={
            "type": "function",
            "function": {"name": "move_arm"}
        },
        temperature=0,
    )
    print(plan)

    current_eef_pose, current_eef_quat = robot.get_ee_pose()
    current_eef_pose = current_eef_pose - movement_bias
    translation_vector = calculate_translation_vector(plan['movement_direction'], plan['movement_amount_centimeters'] / 100)

    # add to history of previous plans to allow replanning
    previous_plans.append(plan)

# get the translation vector for the final outcome
if 'y' == input("Is this OK? (y/n) "):
    move_relative(translation_vector)

camera.close()
