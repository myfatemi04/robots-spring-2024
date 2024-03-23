import os
import pickle

import torch
import torch.utils.data
from demo_to_state_action_pairs import create_torch_dataset


def generate_demos(task_id, n_demos):
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks import OpenDrawer, CloseJar, LightBulbIn, MeatOffGrill
    import time
    
    task_map = {
        "open_drawer": OpenDrawer,
        "close_jar": CloseJar,
        "light_bulb_in": LightBulbIn,
        "meat_off_grill": MeatOffGrill
    }
    
    assert task_id in task_map, f"Task {task_id} not in `task_map`."
    
    data_dir = '/scratch/gsk6me/RLBench_Data/train'
    # n_demos = len(os.listdir(os.path.join(data_dir, task_id)))

    # Get some observations
    env = Environment(
        MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
        data_dir,
        obs_config=ObservationConfig(),
        headless=True)
    env.launch()

    task = env.get_task(task_map[task_id])

    print("Generating demos...")
    
    start_time = time.time()
    
    demos = task.get_demos(n_demos, live_demos=False)
    
    end_time = time.time()

    env.shutdown()
    
    print(f"Finished generating demos. Took {end_time - start_time:.4f} seconds.")

    return demos

def get_demos():
    if not os.path.exists("demos.pkl"):
        with open("demos.pkl", "wb") as f:
            pickle.dump(generate_demos("open_drawer", 8), f)
    else:
        with open("demos.pkl", "rb") as f:
            demos = pickle.load(f)

    return demos

def get_data():
    device = torch.device('cuda')
    dataset = create_torch_dataset(get_demos(), device)

    return dataset
