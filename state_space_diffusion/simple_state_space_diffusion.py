"""
Approach: Similar to RVT.
Generate "virtual views" of a scene. Train a diffusion model to generate future keypoints or grasps with natural language guidance.
Will use a combination of CLIP and Segment Anything features.

Hopefully, should be amenable to massive multi-task learning with high accuracy, using only visual data.

Step 1. Dataset: Samples from RLBench. We will initially only use position, but quickly adopt towards rotation as well.

A case for ``simple" state space diffusion.

"""

# import cv2
import matplotlib.pyplot as plt
import torch

import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import OpenDrawer
from rlbench.observation_config import ObservationConfig

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# from voxel_renderer import VoxelRenderer

# Get some observations
env = Environment(
    MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    '/scratch/gsk6me/RLBench_Data/train',
    obs_config=ObservationConfig(),
    headless=True)
env.launch()

task = env.get_task(OpenDrawer)

print("Getting demos...")
demos = task.get_demos(8, live_demos=False)

env.shutdown()