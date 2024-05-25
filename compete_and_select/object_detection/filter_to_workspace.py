import numpy as np

# We will define the workspace as a box:
min_pt = np.array([0, -0.45, 0])
max_pt = np.array([0.6, 0.45, 1])

def filter_detections_to_workspace(detections, point_cloud_image):
    filtered = []
    for detection in detections:
        box = [int(x) for x in detection.box]
        points_in_detection = point_cloud_image[
            box[1]:box[3],
            box[0]:box[2]
        ].reshape(-1, 3)
        mask = ((points_in_detection > min_pt) & (points_in_detection < max_pt)).all(axis=1)
        if sum(mask) > 0:
            filtered.append(detection)

    return filtered
