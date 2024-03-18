import os
import pickle

import torch
import torch.utils.data
from demo_to_state_action_pairs import create_torch_dataset


def get_demos(n_demos: int = 8):
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks import OpenDrawer

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
    demos = task.get_demos(n_demos, live_demos=False)

    env.shutdown()

    return demos

def get_data():
    if not os.path.exists("demos.pkl"):
        with open("demos.pkl", "wb") as f:
            pickle.dump(get_demos(8), f)
    else:
        with open("demos.pkl", "rb") as f:
            demos = pickle.load(f)

    device = torch.device('cuda')
    dataset = create_torch_dataset(demos, device)

    return dataset
