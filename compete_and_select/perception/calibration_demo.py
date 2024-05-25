import json
import os

import numpy as np
import PIL.Image
from matplotlib import pyplot as plt

from ..panda import Panda
from ..rotation_utils import vector2quat
from .rgbd import RGBD


def run_demo():
    rgbd = RGBD(camera_ids=['000259521012', '000243521012'])

    # Load the extrinsic information
    for i, camera_name in enumerate(['front', 'back_left']):
        with open(os.path.join(os.path.dirname(__file__), f"extrinsics/{camera_name}_camera.json")) as f:
            extrinsics = json.load(f)
        extrinsics = {k: np.array(v) for k, v in extrinsics.items()}

        rgbd.cameras[i].rvec_tvec = (extrinsics['rvec'], extrinsics['translation'])
        rgbd.cameras[i].extrinsic_matrix = np.concatenate([
            extrinsics['rotation_matrix'],
            extrinsics['translation'][:, np.newaxis]
        ], axis=1)

    panda = Panda('192.168.1.222')

    def matplotlib_click_callback(event):
        nonlocal pcds

        x, y = int(event.xdata), int(event.ydata)
        if event.button == 1:
            print(f"Left click at {x}, {y}")
            print(f"Corresponding point: {pcds[0][y, x]}")

            point = pcds[0][y, x].copy()
            point[2] += 0.1
            vertical_rot = vector2quat([0, 0, -1], [0, -1, 0])

            panda.move_to(point, vertical_rot, direct=True)

    # Close the gripper to view precise point
    panda.stop_grasp()
    panda.start_grasp()

    PLOT_MODE = 'rgbd:0'
    # PLOT_MODE = 'point_cloud'
    # This can slow down the program a lot.
    # Applies for RGBD plot mode.
    RGBD_MODE_PLOT_POINT_CLOUDS = True

    if PLOT_MODE.startswith('rgbd:'):
        camera_id = int(PLOT_MODE.split(':')[1])

        # Link the click event listener. We only want to do this in RGBD mode.
        fig = plt.figure()
        fig.canvas.mpl_connect('button_press_event', matplotlib_click_callback)
    
        while True:
            rgbs, pcds = rgbd.capture()

            img = PIL.Image.fromarray(rgbs[camera_id])
            assert pcds[camera_id] is not None, "We imported a calibration, but depth information was still not loaded."

            # Display the image.
            plt.clf()
            plt.axis('off')
            plt.imshow(img)

            if RGBD_MODE_PLOT_POINT_CLOUDS:
                point_cloud_image = pcds[camera_id]
                # Filter out invalid points.
                depth_alpha = (~np.any(point_cloud_image == -10000, axis=-1)).astype(float)
                depth_image = np.linalg.norm(point_cloud_image, axis=-1)
                plt.imshow(depth_image, cmap='viridis', alpha=depth_alpha * 0.2, vmin=0, vmax=2)

            plt.pause(0.05)
    elif PLOT_MODE == 'point_cloud':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Filter to this box.
        min_coords = np.array([0, -1, 0])
        max_coords = np.array([1, 1, 1])
        ax.set_xlim(min_coords[0], max_coords[0])
        ax.set_ylim(min_coords[1], max_coords[1])
        ax.set_zlim(min_coords[2], max_coords[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Skip every N valid points.
        decimate = 50

        while True:
            rgbs, pcds = rgbd.capture()

            ax.clear()
            for i in range(len(rgbd.cameras)):
                if i == 0: continue

                valid_point_mask = np.all((pcds[i] >= min_coords) & (pcds[i] <= max_coords), axis=-1)
                # Should be approximately correct... assuming points are uniformly distributed where they
                # are valid.
                valid_point_mask[::decimate] = False

                # Draw the points.
                ax.scatter(
                    *pcds[i][valid_point_mask].T,
                    c=rgbs[i][valid_point_mask] / 255,
                    s=1
                )

            # Trigger an interruption-free rerender.
            plt.pause(0.05)

if __name__ == '__main__':
    run_demo()
