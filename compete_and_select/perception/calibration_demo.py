import json
import os

import numpy as np
import PIL.Image
from matplotlib import pyplot as plt

from ..panda import Panda
from ..rotation_utils import vector2quat
from .rgbd import RGBD


def run_demo():
    rgbd = RGBD(num_cameras=1)

    # Load the extrinsic information
    with open(os.path.join(os.path.dirname(__file__), "extrinsics/front_camera.json")) as f:
        extrinsics = json.load(f)
    extrinsics = {k: np.array(v) for k, v in extrinsics.items()}

    rgbd.cameras[0].rvec_tvec = (extrinsics['rvec'], extrinsics['translation'])
    rgbd.cameras[0].extrinsic_matrix = np.concatenate([
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

    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', matplotlib_click_callback)

    # Close the gripper to view precise point
    panda.stop_grasp()
    panda.start_grasp()
    
    # This can slow down the program a lot.
    plot_point_clouds = False

    while True:
        rgbs, pcds = rgbd.capture()

        img = PIL.Image.fromarray(rgbs[0])
        assert pcds[0] is not None, "We imported a calibration, but depth information was still not loaded."

        # Display the image.
        plt.axis('off')
        plt.imshow(img)

        if plot_point_clouds:
            point_cloud_image = pcds[0]
            # Filter out invalid points.
            depth_alpha = (~np.any(point_cloud_image == -10000, axis=-1)).astype(float)
            depth_image = np.linalg.norm(point_cloud_image, axis=-1)
            plt.imshow(depth_image, cmap='viridis', alpha=depth_alpha * 0.2, vmin=0, vmax=2)

        plt.pause(0.05)

if __name__ == '__main__':
    run_demo()
