import os
import sys

import cv2
import numpy as np

import apriltag

sys.path.insert(0, "./hand_object_detector")
import detect_hands_pipeline  # type: ignore

os.chdir("hand_object_detector")
detect_hands_pipeline.initialize()
os.chdir("..")

from camera import Camera, triangulate

apriltag_object_points = np.array([
    # order: left bottom, right bottom, right top, left top
    [1, -1/2, 0],
    [1, +1/2, 0],
    [0, +1/2, 0],
    [0, -1/2, 0],
]).astype(np.float32) * 0.1778

class Capture:
    def __init__(self, left: Camera, right: Camera, detect_hands=True, image_width=1280, image_height=720):
        self.left = left
        self.right = right
        self.image_width = image_width
        self.image_height = image_height
        self.apriltag_detector = apriltag.apriltag("tag36h11")
        self.detect_hands = detect_hands

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
        # We may choose to disable hand detection during live recording to improve sampling rate
        if self.detect_hands:
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
