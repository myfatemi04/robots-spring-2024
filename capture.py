import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
import pyk4a

import apriltag

sys.path.insert(0, "./hand_object_detector")
import detect_hands_pipeline  # type: ignore

os.chdir("hand_object_detector")
detect_hands_pipeline.initialize()
os.chdir("..")

import visualization
from camera import Camera, triangulate


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


if pyk4a.connected_device_count() < 2:
    print("Error: Not enough K4A devices connected (<2).")
    exit(1)

k4a_devices = [pyk4a.PyK4A(device_id=i) for i in [0, 1]]
k4a_device_map = {}
for device in k4a_devices:
    device.start()
    k4a_device_map[device.serial] = device

left = Camera(k4a_device_map['000256121012'])
right = Camera(k4a_device_map['000243521012'])

apriltag_object_points = np.array([
    # order: left bottom, right bottom, right top, left top
    [1, -1/2, 0],
    [1, +1/2, 0],
    [0, +1/2, 0],
    [0, -1/2, 0],
]).astype(np.float32) * 0.1778

capture = HandCapture(left, right)
visualizer = visualization.ObjectDetectionVisualizer(live=True)

with open('recordings/recording_003_dummy/camera_calibration.pkl', 'rb') as f:
    calibration = pickle.load(f)
    left.import_calibration(calibration['left'])
    right.import_calibration(calibration['right'])

recording_title = 'open_top_drawer'

# Calculate a serial
recording_serial = 1
existing_recordings = os.listdir('recordings')
while any([f'{recording_serial:03d}' in filename for filename in existing_recordings]):
    recording_serial += 1

prefix = os.path.join('recordings', f'recording_{recording_serial:03d}_{recording_title}')
if not os.path.exists(prefix):
    os.makedirs(prefix)
print("Saving to", prefix)

out_left = cv2.VideoWriter(os.path.join(prefix, 'output_left.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1280, 720))
out_right = cv2.VideoWriter(os.path.join(prefix, 'output_right.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1280, 720))
# Everything in this must be a pure Python object or Numpy array
recording = []

prev_frame_time = time.time()

try:
    while True:
        detection = capture.next()
        left_color = detection['left']['color']
        right_color = detection['right']['color']
        hand_position = detection['hand_position']

        out_left.write(left_color)
        out_right.write(right_color)
        recording.append({
            'timestamp': time.time(),
            'hand_position': hand_position,
        })

        if (hand_position_3d := hand_position['3d']) is not None:
            visualizer.show([
                ('AprilTag', apriltag_object_points.T, 5, (0, 0, 1.0, 1.0)),
                ('Hand', hand_position_3d, 25, (0, 1.0, 0, 1.0))
            ])
            cv2.circle(left_color, hand_position['left_2d'].astype(int), 11, (255, 0, 0), 3)
            cv2.circle(right_color, hand_position['right_2d'].astype(int), 11, (255, 0, 0), 3)

        cv2.imshow('Left camera', left_color)
        cv2.imshow('Right camera', right_color)

        # Print FPS
        frame_time = time.time()
        time_for_frame = (frame_time - prev_frame_time)
        prev_frame_time = frame_time
        print(f"FPS: {1/time_for_frame:.2f}", end='\r')

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    left.close()
    right.close()
    out_left.release()
    out_right.release()

    with open(os.path.join(prefix, "recording.pkl"), "wb") as f:
        pickle.dump(recording, f)

    with open(os.path.join(prefix, "camera_calibration.pkl"), "wb") as f:
        pickle.dump({
            'left': left.export_calibration(),
            'right': right.export_calibration(),
        }, f)
