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

import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh
# Note: We should be able to easily get rid of this dependency. It's just for convenience.
# So far, pytorch3d is ONLY used for the `look_at_view_transform` function.
from pytorch3d.renderer import look_at_view_transform
import torch

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

# Add the fingers
finger_trimesh = trimesh.load_mesh(FINGER_DAE_PATH)
finger_mesh = pyrender.Mesh.from_trimesh([*finger_trimesh.geometry.values()])

def create_hand_node(gripper_width: float = 0.08):
    """
    gripper_width can be from 0 to 0.08.
    """
    # Translate by 0.0584 +Z relative to hand.
    # This is based on franka_description/robots/common/franka_hand.xacro
    finger_z = 0.0584
    finger_matrix_1_relative_to_hand = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, -gripper_width / 2],
        [0, 0, 1, finger_z],
        [0, 0, 0, 1]
    ])
    finger_matrix_2_relative_to_hand = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, +gripper_width / 2],
        [0, 0, 1, finger_z],
        [0, 0, 0, 1]
    ])
    finger_1_node = pyrender.Node(mesh=finger_mesh, matrix=finger_matrix_1_relative_to_hand) # , parent_node=hand_node)
    finger_2_node = pyrender.Node(mesh=finger_mesh, matrix=finger_matrix_2_relative_to_hand) # , parent_node=hand_node)

    hand_node = pyrender.Node(mesh=hand_mesh, children=[finger_1_node, finger_2_node])

    return hand_node

# Stores elev, azim, up.
camera_tfms = {
    "xy": (0, 0, (0, 1, 0),),
    "xz": (90, 0, (0, 0, 1),),
    "yz": (0, 90, (0, 0, 1),),
}

def get_camera(camera_id: str, origin=(0, 0, 0)):
    oc = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)

    # Create vector pointing to origin.
    # distance = 1

    if camera_id == 'xy':
        # R = np.array([
        #     [1, 0, 0],
        #     [0, 1, 0],
        #     [0, 0, -1],
        # ]).T
        # T = np.array([0, 0, -1]) + origin

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
        # R = np.array([
        #     [1, 0, 0],
        #     [0, 0, 1],
        #     [0, -1, 0],
        # ]).T
        # T = np.array([0, -1, 0]) + origin

        backward = np.array([0, -1, 0])
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

        # print(R)
        # print(T)
        # print(origin)

    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = T

    # dist = 1
    # elev, azim, up = camera_tfms[camera_id] # type: ignore
    # elev = torch.tensor([elev])
    # azim = torch.tensor([azim])
    # at = origin # (0, 0, 0)

    # # Must be batched for some reason.
    # r, t = look_at_view_transform(dist=[dist], elev=elev, azim=azim, up=[up], at=[at])
    # r = r[0]
    # t = t[0]

    # matrix = np.eye(4)
    # matrix[:3, :3] = r.T
    # matrix[:3, 3] = r.T @ t

    # print(r, r.T @ t, t)

    return oc, matrix

def render_virtual_plan(gripper_matrix: np.ndarray, origin=(0, 0, 0), gripper_width: float = 0.08):
    # Create the scene.
    scene = pyrender.Scene()
    gripper_node = create_hand_node(gripper_width)
    gripper_node.matrix = gripper_matrix
    scene.add_node(gripper_node)

    # Add ambient light ("Directional Light").
    dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=100.0)
    scene.add(dl)

    # Set background color.
    scene.bg_color = np.array([0.8, 0.8, 1.0])

    # Save to file via offscreen renderer.
    renderer = pyrender.OffscreenRenderer(224, 224)

    # Width and height are inversely proportional to xmag and ymag
    for i, camera_id in enumerate(['xy', 'xz', 'yz']):
        # Create a camera, add it to the scene, and set the main
        # camera node (in place).
        camera, matrix = get_camera(camera_id, origin=origin)
        camera_node = pyrender.Node(camera=camera, matrix=matrix)
        scene.add_node(camera_node)
        scene.main_camera_node = camera_node

        # Oh, cool! We get depth data.
        color, depth = renderer.render(scene) # type: ignore

        plt.subplot(3, 2, i * 2 + 1)
        plt.title("Color " + camera_id)
        plt.imshow(color, origin='lower')
        plt.subplot(3, 2, i * 2 + 2)
        plt.title("Depth")
        plt.imshow(depth, origin='lower')
        
    plt.tight_layout()
    plt.savefig("gripper_rendering_multiview.png", dpi=512)

from get_data import get_demos
from voxel_renderer_slow import SCENE_BOUNDS

demos = get_demos()
demo = demos[0]
obs = demo[0]

gripper_matrix = obs.gripper_matrix
gripper_width = 0.08 * obs.gripper_open

origin = (0.2, 0.0, 1.0)
gripper_t = list(gripper_matrix[:3, 3])

print(gripper_t)

render_virtual_plan(gripper_matrix, origin, gripper_width)
