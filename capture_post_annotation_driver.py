import os
import pickle
import traceback

import cv2
import tqdm

import visualization
from compete_and_select.perception.camera import VirtualCamera
from capture import Capture, apriltag_object_points


def main():
    prefix = 'recordings/recording_011_drawer_play_data_high_sample_rate'
    out_prefix = os.path.join(prefix, "annotated")
    print("Loading from", prefix)

    headless = True

    left = VirtualCamera(os.path.join(prefix, 'output_left.mp4'))
    right = VirtualCamera(os.path.join(prefix, 'output_right.mp4'))

    capture = Capture(left, right, detect_hands=True)
    visualizer = visualization.ObjectDetectionVisualizer(live=True)

    with open('recordings/recording_003_dummy/camera_calibration.pkl', 'rb') as f:
        calibration = pickle.load(f)
        left.import_calibration(calibration['left'])
        right.import_calibration(calibration['right'])

    with open(os.path.join(prefix, 'recording.pkl'), "rb") as f:
        original_recording = pickle.load(f)

    new_recording = []

    print("original recording length:", len(original_recording))

    try:
        for i in tqdm.tqdm(range(len(original_recording)), desc='Annotating...'):
            detection = capture.next()
            left_color = detection['left']['color']
            right_color = detection['right']['color']
            hand_position = detection['hand_position']

            new_recording.append({
                'timestamp': original_recording[i]['timestamp'],
                'hand_position': hand_position,
            })

            if not headless:
                if (hand_position_3d := hand_position['3d']) is not None:
                    visualizer.show([
                        ('AprilTag', apriltag_object_points.T, 5, (0, 0, 1.0, 1.0)),
                        ('Hand', hand_position_3d, 25, (0, 1.0, 0, 1.0))
                    ])
                    cv2.circle(left_color, hand_position['left_2d'].astype(int), 11, (255, 0, 0), 3)
                    cv2.circle(right_color, hand_position['right_2d'].astype(int), 11, (255, 0, 0), 3)

                cv2.imshow('Left camera', left_color)
                cv2.imshow('Right camera', right_color)

            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print("Exception")
        print(e)
        traceback.print_exc()
    finally:
        left.close()
        right.close()

        if not os.path.exists(out_prefix):
            os.makedirs(out_prefix)

        with open(os.path.join(out_prefix, "annotated_recording.pkl"), "wb") as f:
            pickle.dump(new_recording, f)

        with open(os.path.join(out_prefix, "annotated_camera_calibration.pkl"), "wb") as f:
            pickle.dump({
                'left': left.export_calibration(),
                'right': right.export_calibration(),
            }, f)

if __name__ == '__main__':
    main()
