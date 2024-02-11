# Choose a demonstration and identify keypoints
from typing import List
import numpy as np

# https://github.com/stepjam/ARM/blob/main/arm/demo_loading_utils.py

def _is_stopped_old(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = True
    # gripper_state_no_change = (
    #         i < (len(demo) - 2) and
    #         (obs.gripper_open == demo[i + 1].gripper_open and
    #          obs.gripper_open == demo[i - 1].gripper_open and
    #          demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    small_delta = np.allclose(joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def find_keypoints(
    states,
    stopping_delta=0.1
) -> List[int]:
    # We will need to add gripper state detection explicitly
    # at some point.
    episode_keypoints = []
    # prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    nstates = len(states['ee_pos'])
    for i in range(nstates):
        # stopped_buffer requires at least 4 steps of motion before the next keypoint
        small_delta = np.allclose(states['joint_velocities'][i], 0, atol=stopping_delta)
        stopped = (i + 1 < nstates) and small_delta and stopped_buffer <= 0
        if stopped:
            stopped_buffer = 4
        else:
            stopped_buffer -= 1
        
        state_changed = i + 1 == nstates or stopped
        if i != 0 and state_changed:
            episode_keypoints.append(i)
    
    # If the last keypoint was added twice, get rid of it
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    
    return episode_keypoints
