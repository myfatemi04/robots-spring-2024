"""
How do we pick up a cup?

If the cup is vertical, we can just lower ourselves somewhere on the cup's rim.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from ..clip_feature_extraction import get_text_embeds
from ..object_detection.detect_objects import detect
from ..object_detection.object_detection_utils import draw_set_of_marks
from ..panda import Panda
from ..perception.rgbd import RGBD
from ..rotation_utils import vector2quat
from ..segment_point_cloud import SamPointCloudSegmenter

robot = Panda('192.168.1.222')
rgbd = RGBD.autoload('diagonal')
seg = SamPointCloudSegmenter()

robot.robot.go_home()
robot.start_grasp()
robot.stop_grasp()

# some bullshit franka debugging
# robot.move_to(
#     [0.4, 0, 0.2],
#     vector2quat([0, 1, 0], [0, 0, -1]),
#     direct=True
# )

# robot.move_to(
#     [0.4, 0, 0.2],
#     vector2quat([0, 0, -1], [0, -1, 0]),
#     direct=True
# )

# exit()

cup = None

rim_embed = get_text_embeds('a photo of the rim of a cup')[0]

try:
    while True:
        imgs, pcds = rgbd.capture(return_pil=True)

        drawer_handle_detections = detect(imgs[0], 'drawer handle')

        # Visualize the cup detection(s).
        detections_img = draw_set_of_marks(imgs[0], drawer_handle_detections)

        plt.title("Detections")
        plt.imshow(detections_img)
        plt.show()

        # Find something about 1 inch tall
        for detection in drawer_handle_detections:
            point_cloud, color, segs, normal = seg.segment(imgs[0], pcds[0], list(detection.box), imgs[1:], pcds[1:], include_normal_map=True)

            height = np.max(point_cloud[:, 2]) - np.min(point_cloud[:, 2])

            print("Height of drawer handle:", height)

            if height < 0.03:
                selected_point = np.mean(point_cloud, axis=0)
                selected_normal = np.mean(normal, axis=0)
                break
        else:
            print("No drawer handle found.")
            break

        # Project points.
        image_points = cv2.projectPoints(
            selected_point.reshape(1, 3),
            rgbd.cameras[0].rvec_tvec[0],
            rgbd.cameras[0].rvec_tvec[1],
            rgbd.cameras[0].intrinsic_matrix,
            rgbd.cameras[0].distortion_coefficients,
        )[0]
        image_points = image_points.squeeze(1).astype(int)[0]

        # Draw the projected points.
        plt.title("Selected Grasp Location")
        plt.imshow(imgs[0])
        plt.scatter(image_points[0], image_points[1], c='r')
        plt.show()

        horiz_normal = np.array([selected_normal[0], selected_normal[1], 0])
        horiz_normal = horiz_normal / np.linalg.norm(horiz_normal)

        robot.move_to(
            selected_point - (horiz_normal * 0.2),
            vector2quat(horiz_normal, [0, 0, -1]),
            direct=True
        )

        robot.move_to(
            selected_point,
            vector2quat(horiz_normal, [0, 0, -1]),
            direct=True
        )
        robot.start_grasp()

        # Move away from the drawer
        robot.move_by(-horiz_normal * 0.2)

        robot.stop_grasp()

        robot.move_by(-horiz_normal * 0.2)

        break
finally:
    rgbd.close()
