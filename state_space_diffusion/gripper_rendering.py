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

def get_camera(camera_id: str, origin=(0, 0, 0)):
    oc = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)

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
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = T

    return oc, matrix

def render_virtual_plan(gripper_matrix: np.ndarray, origin, points, colors, gripper_width: float = 0.08):
    # Create the scene.
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
    gripper_root_node = create_hand_node(gripper_width)
    # Translate upwards by 0.1 to make the origin be the center between the eef fingers
    eef_correction = -0.11
    gripper_root_node.matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, eef_correction],
        [0, 0, 0, 1],
    ])
    gripper_node_corrected = pyrender.Node(matrix=gripper_matrix, children=[gripper_root_node])
    scene.add_node(gripper_node_corrected)

    # RGBA. Make background transparent.
    scene.bg_color = np.array([0.0, 0.0, 0.0, 1.0])

    # Save to file via offscreen renderer.
    renderer = pyrender.OffscreenRenderer(224, 224)

    point_cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene.add(point_cloud)

    for i, camera_id in enumerate(['xy', 'xz', 'yz']):
        # Create a camera, add it to the scene, and set the main
        # camera node (in place).
        camera, matrix = get_camera(camera_id, origin=origin)
        camera.xmag = 0.5
        camera.ymag = 0.5

        camera_node = pyrender.Node(camera=camera, matrix=matrix)
        scene.add_node(camera_node)
        scene.main_camera_node = camera_node

        # Oh, cool! We get depth data.
        color, depth = renderer.render(scene, pyrender.RenderFlags.RGBA) # type: ignore

        plt.subplot(1, 3, i * 1 + 1)
        plt.title("Color " + camera_id)
        plt.imshow(color, origin='lower')
        
    plt.tight_layout()
    plt.savefig("gripper_rendering_multiview.png", dpi=512)

def test():
    from get_data import get_demos
    from demo_to_state_action_pairs import create_orthographic_labels
    from voxel_renderer_slow import VoxelRenderer, SCENE_BOUNDS
    import torch

    device = 'cuda'

    # TODO: Switch to PyRender for this.
    renderer = VoxelRenderer(SCENE_BOUNDS, 224, torch.tensor([0, 0, 0], device=device), device=device)

    demos = get_demos()
    state_action_tuples = create_orthographic_labels(demos[0], renderer, device, include_extra_metadata=True) # type: ignore

    # Test with first keypoint.
    (yz_image, xz_image, xy_image), (yz_pos, xz_pos, xy_pos), (start_obs, target_obs) = state_action_tuples[2]

    gripper_matrix = start_obs.gripper_matrix
    gripper_width = 0.08 * start_obs.gripper_open

    cameras = [
        'front',
        'left_shoulder',
        'right_shoulder',
        'wrist',
        'overhead',
    ]
    points = [torch.tensor(getattr(start_obs, camera + '_point_cloud').reshape(-1, 3)) for camera in cameras]
    colors = [torch.tensor(getattr(start_obs, camera + '_rgb').reshape(-1, 3) / 255.0) for camera in cameras]
    points = torch.cat(points).numpy()
    colors = torch.cat(colors).numpy()

    mins, maxs = np.array(SCENE_BOUNDS).reshape(2, 3)
    origin = (mins + maxs) / 2

    render_virtual_plan(gripper_matrix, origin, points, colors, gripper_width)

if __name__ == '__main__':
    test()
