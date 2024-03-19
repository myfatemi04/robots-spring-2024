### NOTE: Should run this locally.

"""
Given:
 * Meshes of the "hand" and its fingers
 * A gripper width

Return:
 * A mesh, with the origin being at the midpoint between the two gripper fingers, which can then be rotated
"""

import trimesh
import pyrender
import numpy as np

"""
Requires:
 - trimesh
 - pyrender
 - pycollada
"""

FRANKA_MESH_ROOT = "franka_ros/franka_description/meshes/visual"
HAND_DAE_PATH = f"{FRANKA_MESH_ROOT}/hand.dae"
FINGER_DAE_PATH = f"{FRANKA_MESH_ROOT}/finger.dae"

hand_trimesh = trimesh.load_mesh(HAND_DAE_PATH)
hand_mesh = pyrender.Mesh.from_trimesh([*hand_trimesh.geometry.values()])

scene = pyrender.Scene()
hand_node = scene.add(hand_mesh)

# Add the fingers
finger_trimesh = trimesh.load_mesh(FINGER_DAE_PATH)
finger_mesh = pyrender.Mesh.from_trimesh([*finger_trimesh.geometry.values()])

# Translate by 0.0584 +Z relative to hand
finger_z = 0.0584
gripper_width = 0.08
finger_matrix_1 = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, -gripper_width / 2],
    [0, 0, 1, finger_z],
    [0, 0, 0, 1]
])
finger_matrix_2 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, +gripper_width / 2],
    [0, 0, 1, finger_z],
    [0, 0, 0, 1]
])
finger_1_node = scene.add(finger_mesh, pose=finger_matrix_1, parent_node=hand_node)
finger_2_node = scene.add(finger_mesh, pose=finger_matrix_2, parent_node=hand_node)

pyrender.Viewer(scene, use_raymond_lighting=True)
