import os
import pickle
import time

import cv2
import pyk4a

import visualization
from compete_and_select.perception.camera import Camera
from capture import Capture, apriltag_object_points

def main():
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

if __name__ == '__main__':
    main()
