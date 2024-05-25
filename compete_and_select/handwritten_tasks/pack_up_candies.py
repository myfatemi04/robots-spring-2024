from matplotlib import pyplot as plt

from ..object_detection.detect_objects import detect
from ..object_detection.object_detection_utils import draw_set_of_marks
from ..panda import Panda
from ..perception.rgbd import RGBD
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

        candy_detections = detect(imgs[0], 'a piece of candy')

        # Visualize the candy detections.
        candy_detections_img = draw_set_of_marks(imgs[0], candy_detections)

        plt.title("Candy Detections")
        plt.imshow(candy_detections_img)
        plt.show()

        if len(candy_detections) == 0:
            print("Candies have all been put away.")
            break

        candy = seg.segment_nice(imgs, pcds, candy_detections[0].box)
        x, y, z = candy.centroid
        robot.move_to([x, y, 0.01], direct=True)
        robot.start_grasp()

        robot.move_to([x, y, 0.2], direct=True)
        robot.move_to([cup.centroid[0], cup.centroid[1], 0.2], direct=True)
        robot.stop_grasp()
finally:
    rgbd.close()
