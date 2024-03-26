import numpy as np
import PIL.Image
import quaternions as Q
import torch
import torch.utils.data
from keypoint_generation import get_keypoint_observation_indexes
from rlbench.demo import Demo
from voxel_renderer_slow import VoxelRenderer, SCENE_BOUNDS
from projections import make_projection

# CAMERAS = [
#     'front',
#     'left_shoulder',
#     'right_shoulder',
#     'wrist',
#     'overhead',
# ]

CAMERAS = [
    'front',
    'overhead',
    'left_shoulder',
    'right_shoulder',
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

def scale_image(image: np.ndarray):
    pil_img = PIL.Image.fromarray(image)
    pil_img = pil_img.resize((224, 224), resample=PIL.Image.BILINEAR)
    return np.array(pil_img)

def image_to_tensor(image: np.ndarray, device):
    return torch.tensor(image/255.0, device=device).permute(2, 0, 1)

# v3: uses arbitrary camera matrices
def create_labels_v3(demo: Demo, scale_images=True):
    """
    This is still in Numpy land.
    """
    
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
        intrinsics = [start_obs.misc[camera + '_camera_intrinsics'] for camera in CAMERAS]
        
        if scale_images:
            # Scale intrinsic matrices.
            scale_factor = (224 / 128)
            images = [scale_image(image) for image in images]
            intrinsics = [
                np.array([
                    [intrinsic[0, 0] * scale_factor, 0, intrinsic[0, 2] * scale_factor],
                    [0, intrinsic[1, 1] * scale_factor, intrinsic[1, 2] * scale_factor],
                    [0, 0, intrinsic[2, 2]]
                ])
                for intrinsic in intrinsics
            ]

        pixel_targets = np.stack([
            make_projection(extrinsic, intrinsic, target_pos)
            for extrinsic, intrinsic in zip(extrinsics, intrinsics)
        ])
        
        # Normalize quaternions to each camera, by inverting the camera matrix and applying it to the rotation matrix.
        camera_rotation_matrices = [e[:3, :3] for e in extrinsics]
        quaternion_targets = [
            Q.rotation_matrix_to_quaternion(rotation_matrix.T @ target_rotation_matrix)
            for rotation_matrix in camera_rotation_matrices
        ]

        tuples.append((
            images,
            pixel_targets,
            quaternion_targets,
            {'start_obs': start_obs, 'target_obs': target_obs, 'extrinsics': extrinsics, 'intrinsics': intrinsics}
        ))

        previous_pos = keypoint

    return tuples

def create_torch_dataset_v3(demos, device):
    images = []
    positions = []
    quats = []

    for demo in demos:
        for label in create_labels_v3(demo, scale_images=True):
            images_ = label[0]
            positions_ = label[1]
            quats_ = label[2]
            # extra metadata in label[3]

            images.extend([image_to_tensor(image, device=device) for image in images_])
            positions.extend([torch.tensor(pos, device=device) for pos in positions_])
            quats.extend([torch.tensor(quat, device=device) for quat in quats_])

    images = torch.stack(images)
    positions = torch.stack(positions)
    quats = torch.stack(quats)

    # Create a dataset of images -> 2D positions.
    dataset = torch.utils.data.TensorDataset(images, positions, quats)

    return dataset

# v4: provides access to virtual views *and* true views through the same interface.
def create_labels_v4(demo: Demo, scale_images=True, device='cuda'):
    """
    This is still in Numpy land.
    """
    
    keypoints = get_keypoint_observation_indexes(demo)

    assert keypoints[0] != 0, "Start position is not a keypoint."

    previous_pos = 0

    original_cameras_tuples = []
    virtual_cameras_tuples = []

    renderer = VoxelRenderer(SCENE_BOUNDS, 224, torch.tensor([0, 0, 0], device=device), device)
    virtual_camera_matrices = renderer.get_camera_matrices()

    for keypoint in keypoints:
        # We are predicting KEYPOINT based on our observation at PREVIOUS_POS.
        start_obs = demo[previous_pos]
        target_obs = demo[keypoint]
        target_pos = target_obs.gripper_pose[:3]
        # We use this quaternion, as it is in world frame (instead of gripper_pose[3:])
        target_rotation_matrix = target_obs.gripper_matrix[:3, :3]

        ### Original Cameras ###
        images = [getattr(start_obs, camera + '_rgb') for camera in CAMERAS]
        extrinsics = [start_obs.misc[camera + '_camera_extrinsics'] for camera in CAMERAS]
        intrinsics = [start_obs.misc[camera + '_camera_intrinsics'] for camera in CAMERAS]
        
        if scale_images:
            # Scale intrinsic matrices.
            scale_factor = (224 / 128)
            images = [scale_image(image) for image in images]
            intrinsics = [
                np.array([
                    [intrinsic[0, 0] * scale_factor, 0, intrinsic[0, 2] * scale_factor],
                    [0, intrinsic[1, 1] * scale_factor, intrinsic[1, 2] * scale_factor],
                    [0, 0, intrinsic[2, 2]]
                ])
                for intrinsic in intrinsics
            ]

        pixel_targets = np.stack([
            make_projection(extrinsic, intrinsic, target_pos)
            for extrinsic, intrinsic in zip(extrinsics, intrinsics)
        ])
        
        # Normalize quaternions to each camera, by inverting the camera matrix and applying it to the rotation matrix.
        camera_rotation_matrices = [e[:3, :3] for e in extrinsics]
        quaternion_targets = [
            Q.rotation_matrix_to_quaternion(rotation_matrix.T @ target_rotation_matrix)
            for rotation_matrix in camera_rotation_matrices
        ]

        original_cameras_tuples.append((
            images,
            pixel_targets,
            quaternion_targets,
            {'start_obs': start_obs, 'target_obs': target_obs, 'extrinsics': extrinsics, 'intrinsics': intrinsics, 'projection_types': ['perspective'] * len(extrinsics)}
        ))

        ### Virtual Views ###
        # Get point clouds for virtual views.
        pcds = [torch.tensor(getattr(start_obs, camera + '_point_cloud').reshape(-1, 3)) for camera in CAMERAS]
        colors = [torch.tensor(getattr(start_obs, camera + '_rgb').reshape(-1, 3) / 255.0) for camera in CAMERAS]
        pcd = torch.cat(pcds).to(device)
        color = torch.cat(colors).to(device)
        
        # Render virtual views.
        # x image: axes are (+y, +z)
        # y image: axes are (+x, +z)
        # z image: axes are (+x, +y)
        virtual_images = list(renderer(pcd, color))
        virtual_image_pixel_targets = list(renderer.point_location_on_images(torch.tensor(target_pos, device=device)))
        virtual_rotation_matrices = [extrinsic[:3, :3] for extrinsic, intrinsic in virtual_camera_matrices]
        virtual_image_quaternion_targets = [
            Q.rotation_matrix_to_quaternion(rotation_matrix.T @ target_rotation_matrix)
            for rotation_matrix in virtual_rotation_matrices
        ]
        
        virtual_extrinsics = [e for e, i in virtual_camera_matrices]
        virtual_intrinsics = [i for e, i in virtual_camera_matrices]

        virtual_cameras_tuples.append((
            [(image * 255).cpu().numpy().astype(np.uint8) for image in virtual_images],
            virtual_image_pixel_targets,
            virtual_image_quaternion_targets,
            {
                'start_obs': start_obs,
                'target_obs': target_obs,
                'extrinsics': virtual_extrinsics,
                'intrinsics': virtual_intrinsics,
                'projection_types': ['orthographic'] * len(extrinsics)
            }
        ))

        previous_pos = keypoint

    return original_cameras_tuples, virtual_cameras_tuples

def create_torch_dataset_v4(demos, device, include_original=True, include_virtual=False):
    images = []
    positions = []
    quats = []

    for demo in demos:
        original_tuples, virtual_tuples = create_labels_v4(demo, scale_images=True)
        tuples = []
        if include_original:
            tuples += original_tuples
        if include_virtual:
            tuples += virtual_tuples
        for label in tuples:
            images_ = label[0]
            positions_ = label[1]
            quats_ = label[2]
            # extra metadata in label[3]

            images.extend([image_to_tensor(image, device=device) for image in images_])
            positions.extend([torch.tensor(pos, device=device) for pos in positions_])
            quats.extend([torch.tensor(quat, device=device) for quat in quats_])

    images = torch.stack(images)
    positions = torch.stack(positions)
    quats = torch.stack(quats)

    # Create a dataset of images -> 2D positions.
    dataset = torch.utils.data.TensorDataset(images, positions, quats)

    return dataset

