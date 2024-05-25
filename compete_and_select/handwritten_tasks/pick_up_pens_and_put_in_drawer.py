import dotenv

dotenv.load_dotenv("./compete_and_select/.env")
import time

import numpy as np
from matplotlib import pyplot as plt

from ..object_detection.describe_objects import describe_objects
from ..object_detection.detect_objects import detect
from ..object_detection.filter_to_workspace import \
    filter_detections_to_workspace
from ..object_detection.object_detection_utils import draw_set_of_marks
from ..object_detection_evaluation import standalone_compete_and_select
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

cup = None

try:
    while True:
        imgs, pcds = rgbd.capture(return_pil=True)

        if cup is None:
            cup_detections = detect(imgs[0], 'cup')

            # Visualize the cup detection(s).
            cup_detections_img = draw_set_of_marks(imgs[0], cup_detections)

            plt.title("Cup Detections")
            plt.imshow(cup_detections_img)
            plt.show()

            cup = seg.segment_nice(imgs, pcds, cup_detections[0].box)

        # idea: start with a higher threshold, and decrease the threshold gradually
        # until we find something (or nothing). for example 3-5 detections in a row
        # with "unlikely" or "neutral" the whole time => we stop. this adaptive threshold
        # should prevent unnecessary VLM calls.
        detections = detect(imgs[0], 'dry erase marker', threshold=0.1)

        # filter detections according to reachable area of robot.
        detections = filter_detections_to_workspace(detections, pcds[0])

        print(len(detections), "detections after filtering")

        # Visualize the detections.
        detections_img = draw_set_of_marks(imgs[0], detections)

        plt.title("Detections")
        plt.imshow(detections_img)
        plt.pause(0.05)
        # plt.show()

        if len(detections) == 0:
            print("Pens/pencils have all been put away.")
            break

        boxes = [d.box for d in detections]
        descriptions = describe_objects(imgs[0], boxes)
        print(descriptions)

        RLR = standalone_compete_and_select.select_with_vlm(imgs[0], boxes, "dry erase marker", descriptions, dry_run=False)
        print(RLR['response'])

        index = np.argmax(RLR['logits'])

        if RLR['logits'][index] < 0:
            print("No favorable options left.")
            break

        obj = seg.segment_nice(imgs, pcds, detections[0].box)

        # find longest axis
        axis = obj.point_cloud.max(axis=0) - obj.point_cloud.min(axis=0)
        axis[2] = 0
        axis /= np.linalg.norm(axis)

        up = np.array([0, 0, 1])

        # find a vector perpendicular to the axis
        right = np.cross(axis, up)

        x, y, z = obj.centroid
        robot.move_to(
            [x, y, 0.2],
            vector2quat([0, 0, -1], right),
            direct=True
        )
        robot.move_to(
            [x, y, 0.01],
            vector2quat([0, 0, -1], right),
            direct=True
        )
        robot.start_grasp()

        robot.move_to([x, y, 0.2], direct=True)
        robot.move_to([cup.centroid[0], cup.centroid[1], 0.2], direct=True)
        robot.stop_grasp()

        time.sleep(2)
finally:
    rgbd.close()
