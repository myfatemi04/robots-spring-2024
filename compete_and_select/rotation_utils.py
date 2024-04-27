import numpy as np
from scipy.spatial.transform import Rotation

def vector2quat(claw, right=None):
    claw = claw / np.linalg.norm(claw)
    right = right / np.linalg.norm(right)

    palm = np.cross(right, claw)
    matrix = np.array([
        [palm[0], right[0], claw[0]],
        [palm[1], right[1], claw[1]],
        [palm[2], right[2], claw[2]],
    ])
    
    return Rotation.from_matrix(matrix).as_quat()

def matrix2quat(matrix):
    return Rotation.from_matrix(matrix).as_quat()
