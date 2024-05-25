import numpy as np
from matplotlib import pyplot as plt

from ..object_detection.object_detection_utils import draw_set_of_marks
from ..panda import Panda
from ..perception.rgbd import RGBD
from ..rotation_utils import vector2quat
from ..segment_point_cloud import SamPointCloudSegmenter
from .drawer_api import locate_drawer

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

        drawer = locate_drawer(imgs, pcds)

        # Visualize the detection(s).
        detections_img = draw_set_of_marks(imgs[0], drawer['raw_detections'])

        plt.title("Detections")
        plt.imshow(detections_img)
        plt.show()

        if 'shelf_states' not in drawer:
            print("No drawer found.")
            break

        if max(drawer['shelf_states']) == 0:
            print("All shelves are closed.")
            break

        index = drawer['shelf_states'].index(1)
        selected_point = drawer['shelf_affordance_locations'][index]
        distance_gap = drawer['shelf_depth_gap']
        opening_direction = drawer['opening_direction']

        print("Affordance locations:", drawer['shelf_affordance_locations'])
        print("Opening direction:", opening_direction)
        
        # Project points.
        image_point = rgbd.cameras[0].project_points(selected_point.reshape(1, 3))[0].astype(int)

        # Draw the projected points.
        plt.title("Inferred Open Drawer")
        plt.imshow(imgs[0])
        plt.scatter(image_point[0], image_point[1], c='r')
        plt.show()

        # Close hand to have better ability to push drawer
        robot.start_grasp()

        robot.move_to(
            selected_point + (opening_direction * 0.05),
            vector2quat(-opening_direction, [0, 0, -1]),
            direct=True
        )

        robot.move_to(
            selected_point - (opening_direction * distance_gap),
            vector2quat(-opening_direction, [0, 0, -1]),
            direct=True
        )

        # Move away from the drawer
        robot.move_by(opening_direction * 0.2)

        robot.stop_grasp()

        break
finally:
    rgbd.close()
