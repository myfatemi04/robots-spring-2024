### NOTE: Should run this locally.

"""
Given:
 * Meshes of the "hand" and its fingers
 * A gripper width

Return:
 * A mesh, with the origin being at the midpoint between the two gripper fingers, which can then be rotated
"""

import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import trimesh
import pyrender
import numpy as np

"""
Requires:
 - trimesh
 - pyrender
 - pycollada
"""

FRANKA_MESH_ROOT = "../../franka_ros/franka_description/meshes/visual"
HAND_DAE_PATH = f"{FRANKA_MESH_ROOT}/hand.dae"
FINGER_DAE_PATH = f"{FRANKA_MESH_ROOT}/finger.dae"

hand_trimesh = trimesh.load_mesh(HAND_DAE_PATH)
hand_mesh = pyrender.Mesh.from_trimesh([*hand_trimesh.geometry.values()])

scene = pyrender.Scene()
hand_node = scene.add(hand_mesh)

# Add the fingers
finger_trimesh = trimesh.load_mesh(FINGER_DAE_PATH)
finger_mesh = pyrender.Mesh.from_trimesh([*finger_trimesh.geometry.values()])

# Translate by 0.0584 +Z relative to hand.
# This is based on franka_description/robots/common/franka_hand.xacro
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

# Add ambient light ("Directional Light").
dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=100.0)
scene.add(dl)

# Set background color.
scene.bg_color = np.array([0.8, 0.8, 1.0])

# Width and height are inversely proportional to xmag and ymag
oc = pyrender.OrthographicCamera(xmag=0.2, ymag=0.2)
camera_tfms = [
    # translate
    np.array([
        # The magnitude of the translation does not matter.
        [1, 0, 0, -1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]),
    # Rotate about x/z axes
    np.array([
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]),
    # Rotate about y/z axes
    np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]),
]
scene.add(oc, pose=camera_tfms[0] @ camera_tfms[1] @ camera_tfms[2])

live = False

if live:
    pyrender.Viewer(scene, use_raymond_lighting=True)
else:
    # Save to file via offscreen renderer.
    r = pyrender.OffscreenRenderer(640, 480)
    # Oh, cool! We get depth data.
    color, depth = r.render(scene) # type: ignore

    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    plt.title("Color")
    plt.imshow(color)
    plt.subplot(1, 2, 2)
    plt.title("Depth")
    plt.imshow(depth)
    plt.tight_layout()
    plt.savefig("gripper_rendering.png", dpi=512)
