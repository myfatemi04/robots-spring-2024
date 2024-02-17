import os
import sys

import cv2
import numpy as np

import apriltag
import pyk4a

sys.path.insert(0, "./hand_object_detector")
import detect_hands_pipeline # type: ignore
os.chdir("hand_object_detector")
detect_hands_pipeline.initialize()
os.chdir("..")

from camera import Camera, triangulate
import visualization

if pyk4a.connected_device_count() < 2:
    print("Error: Not enough K4A devices connected (<2).")
    exit(1)

k4a_devices = [pyk4a.PyK4A(device_id=i) for i in [0, 1]]
k4a_device_map = {}
for device in k4a_devices:
    device.start()
    k4a_device_map[device.serial] = device

k4a_left = k4a_device_map['000256121012']
k4a_right = k4a_device_map['000243521012']

apriltag_object_points = np.array([
    # order: left bottom, right bottom, right top, left top
    [1, -1/2, 0],
    [1, +1/2, 0],
    [0, +1/2, 0],
    [0, -1/2, 0],
]).astype(np.float32) * 0.1778

class HandCapture:
    def __init__(self, left: Camera, right: Camera, image_width=1280, image_height=720):
        self.left = left
        self.right = right
        self.image_width = image_width
        self.image_height = image_height
        self.apriltag_detector = apriltag.apriltag("tag36h11")

    def next(self):
        self.left.capture()
        self.right.capture()
        left_color = np.ascontiguousarray(self.left.prev_capture.color[:, :, :3])
        right_color = np.ascontiguousarray(self.right.prev_capture.color[:, :, :3])

        # Ensure that extrinsic matrices are calibrated
        if self.left.extrinsic_matrix is None:
            left_gray = cv2.cvtColor(left_color, cv2.COLOR_RGB2GRAY)
            detections = self.apriltag_detector.detect(left_gray)
            if len(detections) == 1:
                detection = detections[0]
                apriltag_image_points = detection['lb-rb-rt-lt']
                self.left.infer_extrinsics(apriltag_image_points, apriltag_object_points)

        if self.right.extrinsic_matrix is None:
            right_gray = cv2.cvtColor(right_color, cv2.COLOR_RGB2GRAY)
            detections = self.apriltag_detector.detect(right_gray)
            if len(detections) == 1:
                detection = detections[0]
                apriltag_image_points = detection['lb-rb-rt-lt']
                self.right.infer_extrinsics(apriltag_image_points, apriltag_object_points)

        hand_position_3d = None
        hand_position_right_2d = None
        hand_position_left_2d = None
        # Verify that both cameras are calibrated before making hand detections
        if self.left.extrinsic_matrix is not None and self.right.extrinsic_matrix is not None:
            # Perform hand detection
            hand_bounding_boxes_left, _hand_scores = detect_hands_pipeline.detect(left_color)
            if len(hand_bounding_boxes_left) == 1:
                hand_bounding_box = hand_bounding_boxes_left[0]
                x1, y1, x2, y2 = hand_bounding_box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                hand_position_left_2d = np.array([center_x, center_y])

            hand_bounding_boxes_right, _hand_scores = detect_hands_pipeline.detect(right_color)
            if len(hand_bounding_boxes_right) == 1:
                hand_bounding_box = hand_bounding_boxes_right[0]
                x1, y1, x2, y2 = hand_bounding_box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                hand_position_right_2d = np.array([center_x, center_y])
        
            # Triangulate hand position
            if hand_position_right_2d is not None and hand_position_left_2d is not None:
                hand_position_3d = triangulate(self.left, self.right, hand_position_left_2d[None], hand_position_right_2d[None])[0]

        return {
            'left': {
                'color': left_color,
            },
            'right': {
                'color': right_color,
            },
            'hand_position': {
                '3d': hand_position_3d,
                'left_2d': hand_position_left_2d,
                'right_2d': hand_position_right_2d,
            }
        }
        
    def close(self):
        self.left.close()
        self.right.close()

capture = HandCapture(
    Camera(k4a_left),
    Camera(k4a_right),
)
visualizer = visualization.ObjectDetectionVisualizer(live=True)

try:
    while True:
        detection = capture.next()
        left_color = detection['left']['color']
        right_color = detection['right']['color']

        if (hand_position_3d := detection['hand_position']['3d']) is not None:
            visualizer.show([
                ('AprilTag', apriltag_object_points.T, 5, (0, 0, 1.0, 1.0)),
                ('Hand', hand_position_3d, 25, (0, 1.0, 0, 1.0))
            ])
            cv2.circle(left_color, detection['hand_position']['left_2d'].astype(int), 11, (255, 0, 0), 3)
            cv2.circle(right_color, detection['hand_position']['right_2d'].astype(int), 11, (255, 0, 0), 3)

        cv2.imshow('Left camera', left_color)
        cv2.imshow('Right camera', right_color)

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    k4a_left.close()
    k4a_right.close()
