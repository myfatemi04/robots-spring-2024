import numpy as np
import PIL.Image
import quaternions as Q
import torch
import torch.utils.data
from keypoint_generation import get_keypoint_observation_indexes
from rlbench.demo import Demo
from voxel_renderer_slow import VoxelRenderer

CAMERAS = [
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

        pcds = [torch.tensor(getattr(start_obs, camera + '_point_cloud').reshape(-1, 3)) for camera in CAMERAS]
        colors = [torch.tensor(getattr(start_obs, camera + '_rgb').reshape(-1, 3) / 255.0) for camera in CAMERAS]
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
        target_pos = target_obs.gripper_pose[:3]
        # We use this quaternion, as it is in world frame.
        target_quat = Q.rotation_matrix_to_quaternion(target_obs.gripper_matrix[:3, :3])
        # This one has messed up axes for some reason. Maybe it is from the perspective of some other frame.
        # target_quat = target_obs.gripper_pose[3:]

        # Get point clouds for virtual views.
        pcds = [torch.tensor(getattr(start_obs, camera + '_point_cloud').reshape(-1, 3)) for camera in CAMERAS]
        colors = [torch.tensor(getattr(start_obs, camera + '_rgb').reshape(-1, 3) / 255.0) for camera in CAMERAS]
        pcd = torch.cat(pcds).to(device)
        color = torch.cat(colors).to(device)
        
        # Render virtual views.
        # x image: axes are (+y, +z)
        # y image: axes are (+x, +z)
        # z image: axes are (+x, +y)
        (yz_image, xz_image, xy_image) = renderer(pcd, color)
        (yz_pos, xz_pos, xy_pos) = renderer.point_location_on_images(torch.tensor(target_pos, device=device))
        
        yz_quat = Q.compose_quaternions(Q.ROTATE_WORLD_QUATERNION_TO_CAMERA_QUATERNION['yz'], target_quat)
        xz_quat = Q.compose_quaternions(Q.ROTATE_WORLD_QUATERNION_TO_CAMERA_QUATERNION['xz'], target_quat)
        xy_quat = Q.compose_quaternions(Q.ROTATE_WORLD_QUATERNION_TO_CAMERA_QUATERNION['xy'], target_quat)

        tuple_ = ((yz_image, xz_image, xy_image), (yz_pos, xz_pos, xy_pos), (yz_quat, xz_quat, xy_quat))
        if include_extra_metadata:
            tuple_ = (*tuple_, (start_obs, target_obs))
        tuples.append(tuple_)
        
        previous_pos = keypoint

    return tuples

def create_torch_dataset_v2(demos, device):
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
    quats = []

    for demo in demos:
        for (images_, positions_, quats_) in create_orthographic_labels_v2(demo, renderer, device=device):
            # `images_` and `positions_` are sorted into images along x, y, and z axes, respectively.
            # Flip the y axis.
            images.extend([image.permute(2, 0, 1) for image in images_])
            positions.extend([torch.tensor(pos, device=device) for pos in positions_])
            quats.extend([torch.tensor(quat, device=device) for quat in quats_])

    images = torch.stack(images)
    positions = torch.stack(positions)
    quats = torch.stack(quats)

    # Create a dataset of images -> 2D positions.
    dataset = torch.utils.data.TensorDataset(images, positions, quats)

    return dataset

def make_projection(extrinsic, intrinsic, points):
    camera_translation = extrinsic[:3, 3]
    camera_rotation_matrix = extrinsic[:3, :3]

    # [4, 3] -> ([3, 3] @ [3, 4] = [3, 4]).T -> [4, 3]
    pose = (camera_rotation_matrix.T @ (points - camera_translation).T).T
    # [4, 3] -> ([3, 3] @ [3, 4] = [3, 4]).T -> [4, 3]
    pixel_pose_homogeneous = (intrinsic @ pose.T).T
    # Keep the final dimension for broadcasting.
    pixel_pose = pixel_pose_homogeneous[..., :2] / pixel_pose_homogeneous[..., [2]]

    return pixel_pose

CAMERAS = [
    'front',
    'overhead',
    'left_shoulder',
    'right_shoulder',
]

# v3: uses arbitrary camera matrices
def create_labels_v3(demo: Demo):
    keypoints = get_keypoint_observation_indexes(demo)

    assert keypoints[0] != 0, "Start position is not a keypoint."

    previous_pos = 0

    tuples = []

    for keypoint in keypoints:
        # We are predicting KEYPOINT based on our observation at PREVIOUS_POS.
        start_obs = demo[previous_pos]
        target_obs = demo[keypoint]
        target_pos = target_obs.gripper_pose[:3]
        # We use this quaternion, as it is in world frame (instead of gripper_pose[3:])
        target_rotation_matrix = target_obs.gripper_matrix[:3, :3]

        images = [getattr(start_obs, camera + '_rgb') for camera in CAMERAS]
        extrinsics = [start_obs.misc[camera + '_camera_extrinsics'] for camera in CAMERAS]
        camera_rotation_matrices = [e[:3, :3] for e in extrinsics]
        intrinsics = [start_obs.misc[camera + '_camera_intrinsics'] for camera in CAMERAS]

        pixel_targets = [
            make_projection(extrinsic, intrinsic, target_pos)
            for extrinsic, intrinsic in zip(extrinsics, intrinsics)
        ]
        # quaternions normalized to each camera
        quaternion_targets = [
            Q.rotation_matrix_to_quaternion(rotation_matrix.T @ target_rotation_matrix)
            for rotation_matrix in camera_rotation_matrices
        ]

        tuples.append((images, pixel_targets, quaternion_targets, {'start_obs': start_obs, 'target_obs': target_obs, 'extrinsics': extrinsics, 'intrinsics': intrinsics}))

        previous_pos = keypoint

    return tuples

def prepare_image(image: np.ndarray):
    pil_img = PIL.Image.fromarray(image)
    pil_img = pil_img.resize((224, 224), resample=PIL.Image.BILINEAR)
    image = np.array(pil_img)
    tensor = torch.tensor(image/255.0).permute(2, 0, 1)
    return tensor

def create_torch_dataset_v3(demos, device):
    images = []
    positions = []
    quats = []

    for demo in demos:
        for label in create_labels_v3(demo):
            images_ = label[0]
            positions_ = label[1]
            quats_ = label[2]
            # extra metadata in label[3]

            assert images_[0].shape == (128, 128, 3), "Unexpected image shape."

            # Rescale images to 224.
            images.extend([prepare_image(image) for image in images_])
            positions.extend([torch.tensor(pos, device=device) * (224/128) for pos in positions_])
            quats.extend([torch.tensor(quat, device=device) for quat in quats_])

    images = torch.stack(images)
    positions = torch.stack(positions)
    quats = torch.stack(quats)

    # Create a dataset of images -> 2D positions.
    dataset = torch.utils.data.TensorDataset(images, positions, quats)

    return dataset
