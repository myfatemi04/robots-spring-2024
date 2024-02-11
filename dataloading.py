import pickle
import os

import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
import numpy as np

def get_demonstrations_index(base_dir):
    # Extract participants
    rgb_dir = os.path.join(base_dir, "RGB_Data")
    participants = os.listdir(rgb_dir)
    participant_ids = [int(x.split("_")[1]) for x in os.listdir(rgb_dir)]
    
    columns = {
        "Participant": [],
        "Task": [],
        "Interface": [],
        "Trial": [],
    }
    
    for participant_id in sorted(participant_ids):
        participant_dir = os.path.join(rgb_dir, "P_" + str(participant_id))
        available_tasks = os.listdir(participant_dir)
        task_ids = [int(task.split("_")[1]) for task in available_tasks]
        
        for task_id in sorted(task_ids):
            task_dir = os.path.join(participant_dir, "Task_" + str(task_id))
            interfaces = os.listdir(task_dir)
            interface_ids = [int(interface.split("_")[1]) for interface in interfaces]
            
            for interface_id in sorted(interface_ids):
                interface_dir = os.path.join(task_dir, "Interface_" + str(interface_id))
                trials = os.listdir(interface_dir)
                trial_ids = [int(trial.split("_")[1]) for trial in trials]
                
                for trial_id in trial_ids:
                    columns['Participant'].append(participant_id)
                    columns['Task'].append(task_id)
                    columns['Interface'].append(interface_id)
                    columns['Trial'].append(trial_id)
                
    return pd.DataFrame(columns)

def get_observations(base_dir, participant_id, task_id, interface_id, trial_id, plot_sample=False):
    path = os.path.join(base_dir, f"RGB_Data/P_{participant_id}/Task_{task_id}/Interface_{interface_id}/Trial_{trial_id}")
    
    """
    We should have RGBD data from the following sources:
     * kinect{1, 2}_{color, depth}
     * rs_{color, depth}
    
    Along with timestamps for each of them.
    """
    
    # print(sorted(os.listdir(path)))
    
    import pickle
    
    observations = {
        'root': path,
        'data': {},
    }
    
    for i, camera_id in enumerate(['kinect1', 'kinect2', 'rs']):
        ts_file = os.path.join(path, f"{camera_id}_timestamps.pkl" if camera_id != 'rs' else 'realsense_timestamps.pkl')
        
        with open(ts_file, "rb") as f:
            timestamps = pickle.load(f)
            
        # Create file list.
        # Sort by index (which is not the same as sorting by string value)
        color_files = os.listdir(os.path.join(path, f"{camera_id}_color"))
        color_files_sorted = sorted(color_files, key=lambda x: int(x.split("_")[2]))
        
        depth_files = os.listdir(os.path.join(path, f"{camera_id}_depth"))
        depth_files_sorted = sorted(depth_files, key=lambda x: int(x.split("_")[2]))
        
        expected_files = len(timestamps)
        assert len(color_files) == expected_files, f"File count mismatch: {len(color_files)} color_files != {expected_files}"
        assert len(depth_files) == expected_files, f"File count mismatch: {len(depth_files)} depth_files != {expected_files}"
        
        observations['data'][camera_id] = {
            'timestamps': timestamps,
            'color_image_paths': color_files_sorted,
            'depth_image_paths': depth_files_sorted,
        }
        
        if plot_sample:
            color_image_path = os.path.join(path, f"{camera_id}_color", color_files_sorted[0])
            color_image = PIL.Image.open(color_image_path)
            depth_image_path = os.path.join(path, f"{camera_id}_depth", depth_files_sorted[0])
            depth_image = PIL.Image.open(depth_image_path)
            
            plt.subplot(3, 2, i * 2 + 1)
            plt.title(f"Camera {camera_id}: Color image")
            plt.imshow(color_image)
            
            plt.subplot(3, 2, i * 2 + 2)
            plt.title(f"Camera {camera_id}: Depth image")
            plt.imshow(depth_image)
            
    if plot_sample:
        plt.show()
        
    return observations

def get_teleoperation_index(base_dir, participant_id, task_id, interface_id, trial_id):
    teleop_folder = os.path.join(base_dir, f"Joystick_SpaceMouse_Interface/P_{participant_id}/Task_{task_id}/Interface_{interface_id}/Trial_{trial_id}")
    assert os.path.exists(teleop_folder), f"Teleoperation folder does not exist: {teleop_folder}"

    file = os.path.join(teleop_folder, f'robot_data_{participant_id}_{task_id}_{interface_id}_{trial_id}.pkl')
    with open(file, "rb") as f:
        f.seek(0)
        data = pickle.load(f)
    
    return data

def get_vr_data(base_dir, participant_id, task_id, trial_id):
    """
    Loads teleoperation data for the VR interface for the specified
    trial.
    
    File format:
    {
        timestamps: list[float],
        observations: {
            robot_state: {
                ee_pos: float[3]
                ee_quat: float[4]
                ee_pos_vel: float[3]
                ee_ori_vel: float[3]
                joint_positions: float[7]
                joint_velocities: float[7]
                joint_torques: float[7]
                gripper_width: float[1]
            }
        }[]
    }
    
    Interface ID is implied to be 3.
    """
    # Slight typo in path
    teleop_folder = os.path.join(base_dir, f"VR_Interface/Participant_{participant_id}/Task_{task_id}/Interaface_3/Trial_{trial_id}")
    
    file = os.path.join(teleop_folder, f'data.pkl')
    
    assert os.path.exists(file), f"Teleoperation data not found. Path: {teleop_folder}/data.pkl"
    
    with open(file, "rb") as f:
        f.seek(0)
        data = pickle.load(f)
    
    return data

def collate_observations(datum):
    collated = {
        'timestamp': [],
        'ee_pos': [],
        'ee_quat': [],
        'ee_pos_vel': [],
        'ee_ori_vel': [],
        'joint_positions': [],
        'joint_velocities': [],
        'joint_torques': [],
        'gripper_width': [],
    }
    for i in range(len(datum['timestamps'])):
        collated['timestamp'].append(datum['timestamps'][i])
        obs = datum['observations'][i]['robot_state']
        for k in obs:
            collated[k].append(obs[k])
    return {k: np.stack(v, axis=0) for k, v in collated.items()}
    
def plot_demonstration(demonstration, use_timestamps=True):
    plt.figure()
    
    t = demonstration['timestamp']
    t = t - t[0]
    x, y, z = demonstration['ee_pos'].T
    
    xlabel = 'Time (s)'
    
    if not use_timestamps:
        t = np.arange(0, len(t))
        xlabel = 'Step'

    plt.subplot(3, 1, 1)
    plt.title("End-Effector Pose")
    plt.plot(t, x, label='X')
    plt.plot(t, y, label='Y')
    plt.plot(t, z, label='Z')
    plt.xlabel(xlabel)
    plt.ylabel("Position (m)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.title("Gripper Width")
    plt.plot(t, demonstration['gripper_width'])
    plt.xlabel(xlabel)
    plt.ylabel("Width (m)")

    plt.subplot(3, 1, 3)
    plt.title("Joint velocity L2")
    plt.plot(t, np.linalg.norm(demonstration['joint_velocities'], axis=-1))
    plt.xlabel(xlabel)
    plt.ylabel("Velocity (m/s)")

    plt.tight_layout()
    plt.show()
