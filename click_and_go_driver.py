from functools import partial
import os
import pickle
import time

import cv2
import numpy as np
import pyk4a

import visualization
from camera import Camera, triangulate
from capture import Capture, apriltag_object_points

# Required for real-time robot control.
# We may at some point decide to use a different library.
# Therefore, I added this abstraction so exchanging libraries
# here causes libraries to be exchanged everywhere.
import polymetis
import torch

polymetis_server_ip = "192.168.1.222"
robot = polymetis.RobotInterface(
  ip_address=polymetis_server_ip,
  port=50051,
  enforce_version=False,
)
ROBOT_CONTROL_X_BIAS = 0.14
ROBOT_CONTROL_Y_BIAS = 0.03
ROBOT_CONTROL_Z_BIAS = 0.10

def move(x, y, z):
    robot.move_to_ee_pose(torch.tensor([x + ROBOT_CONTROL_X_BIAS, y + ROBOT_CONTROL_Y_BIAS, z + ROBOT_CONTROL_Z_BIAS]).float())

left_pt = None
right_pt = None

def select_pixel(side, event, x, y, flags, param):
    # grab references to the global variables
    global left_pt, right_pt
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONUP:
        if side == 'Left':
            left_pt = (x, y)
        elif side == 'Right':
            right_pt = (x, y)

def main():
    global left_pt, right_pt

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

    capture = Capture(left, right, detect_hands=False)
    visualizer = visualization.ObjectDetectionVisualizer(live=True)

    # Must call this before setMouseCallback to create the window
    cv2.namedWindow("Left")
    cv2.namedWindow("Right")
    # Add callbacks for pixel selection
    cv2.setMouseCallback("Left", partial(select_pixel, "Left"))
    cv2.setMouseCallback("Right", partial(select_pixel, "Right"))

    # Move to approximate center first
    move(0.2, 0.0, 0.2)

    SAVE = False
    if SAVE:
        with open('recordings/recording_003_dummy/camera_calibration.pkl', 'rb') as f:
            calibration = pickle.load(f)
            left.import_calibration(calibration['left'])
            right.import_calibration(calibration['right'])

        recording_title = 'drawer_play_data_high_sample_rate'

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
    target_point = None

    try:
        while True:
            detection = capture.next()
            left_color = detection['left']['color']
            right_color = detection['right']['color']
            hand_position = detection['hand_position']

            if SAVE:
                out_left.write(left_color)
                out_right.write(right_color)
                recording.append({
                    'timestamp': time.time(),
                    'hand_position': hand_position,
                })

            if left_pt is not None:
                cv2.circle(left_color, left_pt, 5, (0, 255, 0), 3)

            if right_pt is not None:
                cv2.circle(right_color, right_pt, 5, (0, 255, 0), 3)

            if target_point is not None:
                go = 'y' == input("Do you wish to move the robot to the selected position? (y/n): ")

                if go:
                    x, y, z = target_point
                    move(x, y, z)

                # Reset the selection after the prompt, regardless of whether we moved or cancelled.
                # If we moved, reset these points.
                # If we cancelled, also reset these points.
                left_pt = None
                right_pt = None
                target_point = None

            if left_pt is not None and right_pt is not None:
                print(left_pt, right_pt)
                target_point = triangulate(left, right, np.array(left_pt)[np.newaxis], np.array(right_pt)[np.newaxis])[0]
            else:
                target_point = None

            visualizer.show([
                ('AprilTag', apriltag_object_points.T, 5, (0.0, 0.0, 1.0, 1.0)),
                *([
                    ('Hand', hand_position['3d'], 25, (0.0, 1.0, 0.0, 1.0))
                ] if hand_position['3d'] is not None else []),
                *([
                    ('Triangulated', target_point, 3, (1.0, 0.0, 0.0, 1.0))
                ] if target_point is not None else [])
            ])
            if hand_position['left_2d']:
                cv2.circle(left_color, hand_position['left_2d'].astype(int), 11, (255, 0, 0), 3)
            if hand_position['right_2d']:
                cv2.circle(right_color, hand_position['right_2d'].astype(int), 11, (255, 0, 0), 3)

            cv2.imshow('Left', left_color)
            cv2.imshow('Right', right_color)

            # Print FPS
            frame_time = time.time()
            time_for_frame = (frame_time - prev_frame_time)
            prev_frame_time = frame_time
            print(f"FPS: {1/time_for_frame:.2f}", end='\r')

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        if SAVE:
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

if __name__ == '__main__':
    main()
