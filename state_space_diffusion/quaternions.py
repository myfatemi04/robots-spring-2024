import numpy as np
from scipy.spatial.transform import Rotation

def create_quaternion(theta, rotation_axis):
    return np.array([np.cos(theta / 2), *(np.sin(theta / 2) * np.array(rotation_axis))])

def compose_quaternions(quat_u, quat_v):
    # standard; real part of uv is -u dot v, and imag. part is u cross v.
    w_u, u = quat_u[0], quat_u[1:]
    w_v, v = quat_v[0], quat_v[1:]
    return np.array([w_u * w_v - np.dot(u, v), *(w_u * v + w_v * u + np.cross(u, v))])

def invert_quaternion(quat):
    # cos(theta/2), sin(theta/2)u -> cos(-theta/2), sin(-theta/2)u = cos(theta/2), -sin(theta/2)u
    return np.array([quat[0], *(-quat[1:])])

### QUATERNIONS FOR XY, XZ, YZ CAMERAS ###
"""
All of these rotations apply to *camera coordinate frame*.
"""
ROTATE_PERSPECTIVE_QUATERNION_TO_WORLD_QUATERNION = {
    # what is in camera x/y is actually in world x/y
    'xy': create_quaternion(0, [1, 0, 0]),
    # what is in camera x/y is actually in world x/z.
    # thus we rotate around camera x by -np.pi/2
    'xz': compose_quaternions(
        create_quaternion(np.pi/2, [0, 1, 0]),
        create_quaternion(np.pi, [0, 0, 1]),
    ),
    # what is in camera x/y is actually in world y/z.
    # thus we rotate camera z up to camera y first (bringin it to world z),
    # then we rotate camera x to world z
    'yz': compose_quaternions(
        create_quaternion(-np.pi/2, [0, 0, 1]), # applied first
        create_quaternion(-np.pi/2, [1, 0, 0]), # applied second
    )
}

ROTATE_WORLD_QUATERNION_TO_CAMERA_QUATERNION = {
    # Inverse of a quaternion can be found by negating sin(theta/2).
    key: invert_quaternion(quat) for key, quat in ROTATE_PERSPECTIVE_QUATERNION_TO_WORLD_QUATERNION.items()
}

def rotation_matrix_to_quaternion(matrix: np.ndarray):
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    else:
        assert matrix.shape == (3, 3), f"Matrix shape is {matrix.shape}. Must be 3x3 (R) or 4x4 (RT)."
    return Rotation.from_matrix(matrix).as_quat()

def quaternion_to_rotation_matrix(quat: np.ndarray):
    # scipy uses scalar-last format
    quat_scalar_last_format = np.array([*quat[1:], quat[0]])
    return Rotation.from_quat(quat_scalar_last_format).as_matrix()
