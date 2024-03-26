import numpy as np
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
)

SCENE_BOUNDS = [
    -0.3,
    -0.5,
    0.6,
    0.7,
    0.5,
    1.6,
]
# [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized

def get_in_bounds_mask(point_cloud, scene_bounds):
    xmin, ymin, zmin, xmax, ymax, zmax = scene_bounds
    return (
        (xmin <= point_cloud[:, 0]) & (point_cloud[:, 0] < xmax) &
        (ymin <= point_cloud[:, 1]) & (point_cloud[:, 1] < ymax) &
        (zmin <= point_cloud[:, 2]) & (point_cloud[:, 2] < zmax)
    )

def get_camera_matrices(camera_id: str, origin=(0, 0, 0)):
    # Create vector pointing to origin.
    # See https://learnopengl.com/Getting-started/Camera.
    if camera_id == 'xy':
        backward = np.array([0, 0, 1])
        right = np.array([1, 0, 0])
        up = np.cross(backward, right)
        
        R = np.array([
            right,
            up,
            backward,
        ]).T
        T = (backward + origin)
    elif camera_id == 'xz':
        backward = np.array([0, 1, 0])
        right = np.array([1, 0, 0])
        up = np.cross(backward, right)
        
        R = np.array([
            right,
            up,
            backward,
        ]).T
        T = (backward + origin)
    elif camera_id == 'yz':
        backward = np.array([-1, 0, 0])
        right = np.array([0, 1, 0])
        up = np.cross(backward, right)
        
        R = np.array([
            right,
            up,
            backward,
        ]).T
        T = (backward + origin)
    
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = T

    image_size = 224
    intrinsic = np.array([
        [image_size / 2, 0, image_size / 2],
        [0, image_size / 2, image_size / 2],
        [0, 0, 1]
    ])

    return extrinsic, intrinsic

def render_scene():
    pass
