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
from ..perception.rgbd import RGBD, get_normal_map
from ..rotation_utils import vector2quat
from ..segment_point_cloud import SamPointCloudSegmenter

robot = Panda('192.168.1.222')
rgbd = RGBD.autoload('diagonal')
seg = SamPointCloudSegmenter()

robot.robot.go_home()
robot.start_grasp()
robot.stop_grasp()

cup = None

rim_embed = get_text_embeds('a photo of the rim of a cup')[0]

try:
    while True:
        imgs, pcds = rgbd.capture(return_pil=True)

        cup_detections = detect(imgs[0], 'cup')

        # Visualize the cup detection(s).
        cup_detections_img = draw_set_of_marks(imgs[0], cup_detections)

        plt.title("Cup Detections")
        plt.imshow(cup_detections_img)
        plt.show()

        cup_detection = cup_detections[1]
        point_cloud, color, seg, normal = seg.segment(imgs[0], pcds[0], list(cup_detection.box), imgs[1:], pcds[1:], include_normal_map=True)

        ##### Begin unhinged experiment to segment cup rim in point cloud space using CLIP ###
        be_unhinged = False
        if be_unhinged:
            cup = seg.segment_nice_clip(imgs, pcds, cup_detection.box)

            # Get the "rim" of the cup (through CLIP)
            point_scores = cup.clip_features @ rim_embed

            # Draw the points on the cup, highlighted by the CLIP match score
            point_scores = (point_scores - point_scores.min()) / (point_scores.max() - point_scores.min())

            # Project points.
            image_points = cv2.projectPoints(
                cup.point_cloud,
                rgbd.cameras[0].rvec_tvec[0],
                rgbd.cameras[0].rvec_tvec[1],
                rgbd.cameras[0].intrinsic_matrix,
                rgbd.cameras[0].distortion_coefficients,
            )[0]
            image_points = image_points.squeeze(1).astype(int)
            cmap = plt.get_cmap('coolwarm')
            colors = np.array(cmap(point_scores))[:, :3]
            drawn_image = np.array(imgs[0])
            drawn_image[image_points[:, 1], image_points[:, 0]] = (colors * 255).astype(int)
            # Draw the object affordance locations.
            plt.imshow(drawn_image)
            plt.show()

        # Get the top 1 cm of the cup
        top_z = point_cloud[:, 2].max()
        top_points_mask = (
            (point_cloud[:, 2] < (top_z - 0.01)) &
            (point_cloud[:, 2] > (top_z - 0.02))
        )
        top_points = point_cloud[top_points_mask]
        top_points_normals = normal[top_points_mask]

        index = np.random.randint(0, len(top_points))
        selected_point = top_points[index]
        selected_normal = top_points_normals[index]

        robot.move_to(
            selected_point,
            vector2quat([0, 0, -1], [selected_normal[0], selected_normal[1], 0])
        )
        robot.start_grasp()

        # Lift up the cup
        robot.move_by([0, 0, 0.3])

        break
finally:
    rgbd.close()
