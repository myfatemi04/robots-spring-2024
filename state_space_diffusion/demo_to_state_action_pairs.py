from keypoint_generation import get_keypoint_observation_indexes
from rlbench.demo import Demo
from voxel_renderer_slow import VoxelRenderer
import torch
import torch.utils.data

cameras = [
    'front',
    'left_shoulder',
    'right_shoulder',
    'wrist',
    'overhead',
]

def create_orthographic_labels(demo: Demo, renderer: VoxelRenderer, device="cuda", include_extra_metadata=False):
    keypoints = get_keypoint_observation_indexes(demo)


    assert keypoints[0] != 0, "Start position is not a keypoint."

    previous_pos = 0

    tuples = []

    for keypoint in keypoints:
        # We are predicting KEYPOINT based on our observation at PREVIOUS_POS.
        start_obs = demo[previous_pos]
        target_obs = demo[keypoint]
        eef_pos = target_obs.gripper_pose[:3]

        pcds = [torch.tensor(getattr(start_obs, camera + '_point_cloud').reshape(-1, 3)) for camera in cameras]
        colors = [torch.tensor(getattr(start_obs, camera + '_rgb').reshape(-1, 3) / 255.0) for camera in cameras]
        pcd = torch.cat(pcds).to(device)
        color = torch.cat(colors).to(device)
        
        # Create a state-action tuple.
        # x image: axes are (+y, +z)
        # y image: axes are (+x, +z)
        # z image: axes are (+x, +y)
        (x_image, y_image, z_image) = renderer(pcd, color)
        (x_pos, y_pos, z_pos) = renderer.point_location_on_images(torch.tensor(eef_pos, device=device))

        if include_extra_metadata:
            tuples.append(((x_image, y_image, z_image), (x_pos, y_pos, z_pos), (start_obs, target_obs)))
        else:
            # for compatibility
            tuples.append(((x_image, y_image, z_image), (x_pos, y_pos, z_pos)))
        
        previous_pos = keypoint

    return tuples

def create_torch_dataset(demos, device):
    SCENE_BOUNDS = [
        -0.3, -0.5,
        0.6, 0.7,
        0.5, 1.6,
    ]
    VOXEL_IMAGE_SIZE = 224
    BACKGROUND_COLOR = torch.tensor([0, 0, 0], device=device)
    # Use this to generate the input images.
    renderer = VoxelRenderer(SCENE_BOUNDS, VOXEL_IMAGE_SIZE, BACKGROUND_COLOR, device=device)

    images = []
    positions = []

    for demo in demos:
        for (images_, positions_) in create_orthographic_labels(demo, renderer, device=device):
            # `images_` and `positions_` are sorted into images along x, y, and z axes, respectively.
            # Flip the y axis.
            images.extend([image.permute(2, 0, 1) for image in images_])
            positions.extend([torch.tensor(pos, device=device) for pos in positions_])

    images = torch.stack(images)
    positions = torch.stack(positions)

    # Create a dataset of images -> 2D positions.
    dataset = torch.utils.data.TensorDataset(images, positions)

    return dataset

# v2: includes quaternions
def create_orthographic_labels_v2(demo: Demo, renderer: VoxelRenderer, device="cuda", include_extra_metadata=False):
    keypoints = get_keypoint_observation_indexes(demo)


    assert keypoints[0] != 0, "Start position is not a keypoint."

    previous_pos = 0

    tuples = []

    for keypoint in keypoints:
        # We are predicting KEYPOINT based on our observation at PREVIOUS_POS.
        start_obs = demo[previous_pos]
        target_obs = demo[keypoint]
        eef_pos = target_obs.gripper_pose[:3]

        # Get point clouds for virtual views.
        pcds = [torch.tensor(getattr(start_obs, camera + '_point_cloud').reshape(-1, 3)) for camera in cameras]
        colors = [torch.tensor(getattr(start_obs, camera + '_rgb').reshape(-1, 3) / 255.0) for camera in cameras]
        pcd = torch.cat(pcds).to(device)
        color = torch.cat(colors).to(device)
        
        # Render virtual views.
        # x image: axes are (+y, +z)
        # y image: axes are (+x, +z)
        # z image: axes are (+x, +y)
        (yz_image, xz_image, xy_image) = renderer(pcd, color)
        (yz_pos, xz_pos, xy_pos) = renderer.point_location_on_images(torch.tensor(eef_pos, device=device))

        if include_extra_metadata:
            tuples.append(((yz_image, xz_image, xy_image), (yz_pos, xz_pos, xy_pos), (start_obs, target_obs)))
        else:
            # for compatibility
            tuples.append(((yz_image, xz_image, xy_image), (yz_pos, xz_pos, xy_pos)))
        
        previous_pos = keypoint

    return tuples
