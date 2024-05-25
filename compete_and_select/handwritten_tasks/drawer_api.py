from matplotlib import pyplot as plt
import numpy as np

from ..object_detection.detect_objects import detect
from ..segment_point_cloud import SamPointCloudSegmenter

seg = SamPointCloudSegmenter()

def locate_drawer(imgs, pcds):
    drawer_shelf_detections = detect(imgs[0], 'drawer handle', threshold=0.05)
    drawer_open_direction = None
    shelf_affordance_locations = []

    # Sort from top to bottom
    for i, detection in enumerate(
        sorted(drawer_shelf_detections, key=lambda x: (x.box[1] + x.box[3]) / 2)
    ):
        point_cloud, color, segs, normal = seg.segment(imgs[0], pcds[0], list(detection.box), imgs[1:], pcds[1:], include_normal_map=True)

        if len(point_cloud) == 0:
            continue

        # Visualize the point cloud.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Point Cloud: " + str(i + 1))
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        plt.show()

        height = np.max(point_cloud[:, 2]) - np.min(point_cloud[:, 2])

        print("Drawer handle candidate", i + 1)
        print("Max Z:", np.max(point_cloud[:, 2]))
        print("Min Z:", np.min(point_cloud[:, 2]))
        print("Height:", height)

        if height < 0.03:
            shelf_affordance_locations.append(np.mean(point_cloud, axis=0))
            drawer_open_direction = -np.mean(normal, axis=0)

    if drawer_open_direction is None:
        print("No drawer handle found.")
        return {'raw_detections': drawer_shelf_detections}

    drawer_open_direction[2] = 0
    drawer_open_direction = drawer_open_direction / np.linalg.norm(drawer_open_direction)
    
    # Measure which point is "further along" the normal.
    # (There are two other orthogonal components to this normal)
    print(drawer_open_direction)

    distances = [pt @ drawer_open_direction for pt in shelf_affordance_locations]

    gap = max(distances) - min(distances)

    # 0 = closed, 1 = open
    shelf_states = [0] * len(shelf_affordance_locations)

    if gap > 0.08:
        shelf_states[np.argmax(distances)] = 1

    return {
        "raw_detections": drawer_shelf_detections,
        "opening_direction": drawer_open_direction,
        "shelf_affordance_locations": shelf_affordance_locations,
        "shelf_states": shelf_states,
        "shelf_depth_gap": gap,
    }
