from keypoint_generation import get_keypoint_observation_indexes
from rlbench.demo import Demo
from voxel_renderer_slow import VoxelRenderer
import torch

cameras = [
    'front',
    'left_shoulder',
    'right_shoulder',
    'wrist',
    'overhead',
]

def create_orthographic_labels(demo: Demo, renderer: VoxelRenderer, device="cuda"):
    keypoints = get_keypoint_observation_indexes(demo)


    assert keypoints[0] != 0, "Start position is not a keypoint."

    previous_pos = 0

    tuples = []

    for keypoint in keypoints:
        # We are predicting KEYPOINT based on our observation at PREVIOUS_POS.
        obs = demo[previous_pos]
        eef_pos = demo[keypoint].gripper_pose[:3]

        pcds = [torch.tensor(getattr(obs, camera + '_point_cloud').reshape(-1, 3)) for camera in cameras]
        colors = [torch.tensor(getattr(obs, camera + '_rgb').reshape(-1, 3) / 255.0) for camera in cameras]
        pcd = torch.cat(pcds).to(device)
        color = torch.cat(colors).to(device)
        
        # Create a state-action tuple.
        # x image: axes are +z and +y.
        # y image: axes are +z and +x.
        # z image: axes are +x and +y.
        (x_image, y_image, z_image) = renderer(pcd, color)
        (x_pos, y_pos, z_pos) = renderer.point_location_on_images(torch.tensor(eef_pos, device=device))

        tuples.append(((x_image, y_image, z_image), (x_pos, y_pos, z_pos)))
        
        previous_pos = keypoint

    return tuples
