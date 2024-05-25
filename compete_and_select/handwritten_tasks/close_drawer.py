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

print(":: Going home ::")
robot.robot.go_home()

print(":: Opening and closing gripper ::")
robot.start_grasp()
robot.stop_grasp()

try:
    while True:
        imgs, pcds = rgbd.capture(return_pil=True)

        drawer_handle_detections = detect(imgs[0], 'drawer handle', threshold=0.05)

        # Visualize the cup detection(s).
        detections_img = draw_set_of_marks(imgs[0], drawer_handle_detections)

        plt.title("Detections")
        plt.imshow(detections_img)
        plt.show()

        reference_normal = None
        points = []

        for detection in drawer_handle_detections:
            point_cloud, color, segs, normal = seg.segment(imgs[0], pcds[0], list(detection.box), imgs[1:], pcds[1:], include_normal_map=True)

            if len(point_cloud) == 0:
                continue

            height = np.max(point_cloud[:, 2]) - np.min(point_cloud[:, 2])

            print("Height of drawer handle:", height)

            if height < 0.03:
                points.append(np.mean(point_cloud, axis=0))
                reference_normal = -np.mean(normal, axis=0)
                
        if reference_normal is None:
            print("No drawer handle found.")
            continue

        reference_normal[2] = 0
        reference_normal = reference_normal / np.linalg.norm(reference_normal)
        
        # Measure which point is "further along" the normal.
        # (There are two other orthogonal components to this normal)
        print(reference_normal)

        distances = [pt @ reference_normal for pt in points]
        selected_point = points[np.argmax(distances)]

        print(distances)

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
        plt.title("Inferred Open Drawer")
        plt.imshow(imgs[0])
        plt.scatter(image_points[0], image_points[1], c='r')
        plt.show()

        distance_gap = max(distances) - min(distances)

        robot.start_grasp()

        robot.move_to(
            selected_point + (reference_normal * 0.05),
            vector2quat(-reference_normal, [0, 0, -1]),
            direct=True
        )

        robot.move_to(
            selected_point - (reference_normal * distance_gap),
            vector2quat(-reference_normal, [0, 0, -1]),
            direct=True
        )

        # Move away from the drawer
        robot.move_by(reference_normal * 0.2)

        robot.stop_grasp()

        break
finally:
    rgbd.close()
