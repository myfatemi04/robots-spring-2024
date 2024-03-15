import pickle
import os
import PIL.Image

import torch
import torch.utils.data
import numpy as np

import keypoint_generation

# Get keypoint position on image.
# From https://github.com/stepjam/PyRep/blob/dev/pyrep/objects/vision_sensor.py#L168
def get_camera_projection_matrix(intrinsics, extrinsics):
    # Get 3x3 rotation matrix
    R = extrinsics[:3, :3]
    # Get 3x1 translation matrix
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    # inverse of rotation matrix is transpose, as dimensions are orthonormal.
    # Invert C
    R_inv = R.T
    R_inv_C = np.matmul(R_inv, C)
    # Create new matrix
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    # Multiply matrices to form camera projection matrix
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    return cam_proj_mat

def gripper_pose_to_pixel(pos, camera_matrix):
    x, y, z = pos[:3]
    # The 1 is a buffer dimension
    pixel_x, pixel_y, pixel_homo = camera_matrix @ np.array([x, y, z, 1])
    # Normalize by the buffer dimension
    return (pixel_x / pixel_homo, pixel_y / pixel_homo)

def get_low_dim_observations(episode_path: str):
    """
    Note that if we load `low_dim_obs.pkl`, we can save a ton of time by skipping initial image loading.
    We can load corresponding images lazily from {episode_path}/{image_topic_name}/{image_index}.png
    """
    with open(os.path.join(episode_path, "low_dim_obs.pkl"), "rb") as f:
        demo = pickle.load(f)
    return demo

def get_flat_keypoint_dataset(episode_paths: list):
    """
    Requires RLBench.
    """
    results = []
    for path in episode_paths:
        demo = get_low_dim_observations(path)
        keypoint_observation_indexes = keypoint_generation.get_keypoint_observation_indexes(demo)
        previous_keypoint = -1
        for index in keypoint_observation_indexes:
            # obs[previous_keypoint + 1] targets obs[index]
            results.append((path, demo, previous_keypoint + 1, index))
            previous_keypoint = index
    return results

cameras = [
    'front',
    'wrist',
    'overhead',
    'right_shoulder',
    'left_shoulder',
]

class ObsToKeypointDataset(torch.utils.data.Dataset):
    """
    Assumes keypoints have already been detected.
    dataset(episode_path, keypoints): index -> (obs, xyz keypoint location)
    achieves this by flattening episode paths.
    """
    def __init__(self, flat_dataset):
        self.flat_dataset = flat_dataset
    
    @staticmethod
    def from_folder(folder: str):
        paths = [
            os.path.join(folder, episode) for episode in os.listdir(folder)
        ]
        flat_dataset = get_flat_keypoint_dataset(paths)
        return ObsToKeypointDataset(flat_dataset)
        
    def __getitem__(self, index: int):
        path, demo, observation_index, target_keypoint_index = self.flat_dataset[index]
        # Populate images
        target_keypoint_obs = demo[target_keypoint_index]
        gripper_x, gripper_y, gripper_z = target_keypoint_obs.gripper_pose[:3]
        # Get target location on all cameras
        return ({
            "images": {
                camera: PIL.Image.open(os.path.join(path, camera + "_rgb", str(observation_index) + ".png")) for camera in cameras
            }
        }, {
            "gripper_pose_3d": (gripper_x, gripper_y, gripper_z),
            "gripper_pose_2d": {
                camera: gripper_pose_to_pixel(target_keypoint_obs.gripper_pose, get_camera_projection_matrix(
                    target_keypoint_obs.misc[camera + "_camera_intrinsics"],
                    target_keypoint_obs.misc[camera + "_camera_extrinsics"],
                )) for camera in cameras
            },
        })

    def __len__(self):
        return len(self.flat_dataset)

if __name__ == '__main__':
    import logging
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info("Loading data...")

    example_episodes_root = "/scratch/gsk6me/RLBench_Data/train/open_drawer/variation0/episodes/"
    # example_episodes = [
    #     os.path.join(example_episodes_root, episode) for episode in os.listdir(example_episodes_root)
    # ]

    dataset = ObsToKeypointDataset.from_folder(example_episodes_root)
    obs, target = dataset[1]

    plt.clf()
    for i in range(len(cameras)):
        camera = cameras[i]
        image = np.array(obs["images"][camera])
        plt.subplot(2, 3, i + 1)
        plt.title("Camera " + camera)
        plt.imshow(image)
        x, y = target["gripper_pose_2d"][camera]
        plt.scatter(x, y)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("test.png")

    logging.info("Generated example data: %d keypoints", len(dataset))

    # logging.info("Loaded data. Detecting keypoints...")

    # get_flat_keypoint_dataset(example_episodes)

    # logging.info("Detected keypoints. Creating dataset...")