import time
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import cv2

from camera import VirtualCamera

# Slight code duplication but whatever
class CameraSetup:
    def __init__(self, rvec_tvec: np.ndarray, extrinsic_matrix: np.ndarray, intrinsic_matrix: np.ndarray, distortion_coefficients: np.ndarray):
        self.rvec_tvec = rvec_tvec
        self.extrinsic_matrix = extrinsic_matrix
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coefficients = distortion_coefficients
        
    def project_points(self, object_points):
        assert self.extrinsic_matrix is not None
        
        rvec, tvec = self.rvec_tvec
        points, _jacobian = cv2.projectPoints(object_points, rvec, tvec, self.intrinsic_matrix, self.distance_coefficients)
        points = points[:, 0, :]

        return points
    
    # Useful for tasks where the AprilTag starts off occluded.
    def export_calibration(self):
        return {
            'rvec_tvec': self.rvec_tvec,
            'extrinsic_matrix': self.extrinsic_matrix,
            'intrinsic_matrix': self.intrinsic_matrix,
            'distortion_coefficients': self.distortion_coefficients,
        }

    def import_calibration(self, calibration):
        self.rvec_tvec = calibration.get('rvec_tvec', None)
        self.extrinsic_matrix = calibration.get('extrinsic_matrix', None)
        self.intrinsic_matrix = calibration.get('intrinsic_matrix', None)
        self.distortion_coefficients = calibration.get('distortion_coefficients', None)
        self.original_calibration = False
    
    def undistort(self, image_points):
        # Note: This undoes the intrinsic matrix as well
        return cv2.undistortPoints(image_points.astype(np.float32), self.intrinsic_matrix, self.distortion_coefficients)

# Render a plan virtually.
def render_virtual_plan(camera_setups: List[VirtualCamera], images: np.ndarray, plan_xyz: np.ndarray):
    results = np.zeros((len(plan_xyz), len(camera_setups), 720, 1280, 3))
    for i, setup in enumerate(camera_setups):
        points = setup.project_points(plan_xyz)
        results[:, i, :, :, :] = images[i]
        for j, (x, y) in enumerate(points):
            cv2.circle(results[j, i], (int(x), int(y)), 25, (0, 255, 0), -1)

    return results

def main():
    import pickle
    from generate_horizon_dataset import smoothen_with_spline
    
    with open("recordings/recording_011_drawer_play_data_high_sample_rate/annotated/annotated_recording.pkl", "rb") as f:
        recording = pickle.load(f)

    visualize = False
    dataset = []

    left = VirtualCamera('recordings/recording_011_drawer_play_data_high_sample_rate/output_left.mp4')
    right = VirtualCamera('recordings/recording_011_drawer_play_data_high_sample_rate/output_right.mp4')

    with open('recordings/recording_011_drawer_play_data_high_sample_rate/camera_calibration.pkl', 'rb') as f:
        calibration = pickle.load(f)
        left.import_calibration(calibration['left'])
        right.import_calibration(calibration['right'])
    
    horizon_frames = 40
    speed = 2
    prev_timestamp = recording[0]['timestamp']
    for i in range(1, len(recording)):
        step = recording[i]
        # Simulate realistic playback speed
        if visualize:
            time.sleep((step['timestamp'] - prev_timestamp) / speed)
        prev_timestamp = step['timestamp']

        capture_left = left.capture()
        capture_right = right.capture()

        times = []
        X = []
        Y = []
        Z = []

        for j in range(i, min(i + horizon_frames, len(recording))):
            pos = recording[j]['hand_position']['3d']
            if pos is not None:
                x, y, z = pos
                times.append(recording[j]['timestamp'])
                X.append(x)
                Y.append(y)
                Z.append(z)

        if len(times) < 0.8 * horizon_frames:
            continue

        times = np.array(times)
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        times_smooth, X_smooth = smoothen_with_spline(times, X)
        _, Y_smooth = smoothen_with_spline(times, Y)
        _, Z_smooth = smoothen_with_spline(times, Z)

        dataset.append((times_smooth, X_smooth, Y_smooth, Z_smooth))

        rendering = render_virtual_plan([left, right], np.stack((capture_left.color, capture_right.color), axis=0), np.stack((X, Y, Z), axis=1))

        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.title(f"left: i={i+1}/5")
            plt.imshow(rendering[int((i/5) * rendering.shape[0]), 0] / 255.0)

            plt.subplot(2, 5, i + 1 + 5)
            plt.title(f"right: i={i+1}/5")
            plt.imshow(rendering[int((i/5) * rendering.shape[0]), 1] / 255.0)

        plt.tight_layout()
        plt.show()
        exit()

    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
