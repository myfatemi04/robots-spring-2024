from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import PIL.Image
import pyk4a


class Camera:
    def __init__(self, k4a: pyk4a.PyK4A):
        self.k4a = k4a
        # Camera parameters
        self.rvec_tvec = None
        self.extrinsic_matrix = None
        self.intrinsic_matrix = k4a.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
        self.distortion_coefficients = k4a.calibration.get_distortion_coefficients(pyk4a.CalibrationType.COLOR)
        self.original_calibration = True
        self.prev_capture: Optional[pyk4a.PyK4ACapture] = None

    def capture(self):
        self.prev_capture = self.k4a.get_capture()
        return self.prev_capture

    def infer_extrinsics(self, apriltag_points, apriltag_object_points):
        ret, rvec, tvec = cv2.solvePnP(apriltag_object_points, apriltag_points, self.intrinsic_matrix, self.distortion_coefficients)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation = tvec
        extrinsics = np.concatenate((rotation_matrix, translation), axis=1)

        self.rvec_tvec = (rvec, tvec)
        self.extrinsic_matrix = extrinsics

    def project_points(self, object_points):
        assert self.extrinsic_matrix is not None
        
        rvec, tvec = self.rvec_tvec
        points, _jacobian = cv2.projectPoints(object_points, rvec, tvec, self.intrinsic_matrix, self.distortion_coefficients)
        points = points[:, 0, :]

        return points
    
    def undistort(self, image_points):
        # Note: This undoes the intrinsic matrix as well
        return cv2.undistortPoints(image_points.astype(np.float32), self.intrinsic_matrix, self.distortion_coefficients)
    
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

    def close(self):
        self.k4a.close()

@dataclass
class VirtualCapture:
    color: np.ndarray

# For simulating playback on a video file.
# If `filename` is .png, we just return that .png indefinitely...
class VirtualCamera:
    def __init__(self, filename):
        self.filename = filename
        if filename.endswith(".png"):
            self.image = np.array(PIL.Image.open(filename))
            self.video_capture = None
        else:
            self.image = None
            self.video_capture = cv2.VideoCapture(filename)
        # Camera parameters
        self.rvec_tvec = None
        self.extrinsic_matrix = None
        self.intrinsic_matrix = None
        self.distortion_coefficients = None
        self.original_calibration = True
        self.prev_capture: Optional[pyk4a.PyK4ACapture] = None

    def capture(self):
        if self.image is not None:
            self.prev_capture = VirtualCapture(color=self.image)
        else:
            _, image_color = self.video_capture.read()
            self.prev_capture = VirtualCapture(color=image_color)
        return self.prev_capture

    def infer_extrinsics(self, apriltag_points, apriltag_object_points):
        ret, rvec, tvec = cv2.solvePnP(apriltag_object_points, apriltag_points, self.intrinsic_matrix, self.distortion_coefficients)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation = tvec
        extrinsics = np.concatenate((rotation_matrix, translation), axis=1)

        self.rvec_tvec = (rvec, tvec)
        self.extrinsic_matrix = extrinsics

    def project_points(self, object_points):
        assert self.extrinsic_matrix is not None
        
        rvec, tvec = self.rvec_tvec
        points, _jacobian = cv2.projectPoints(object_points, rvec, tvec, self.intrinsic_matrix, self.distortion_coefficients)
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
    
    def close(self):
        self.video_capture.release()

def triangulate(camera1: Camera, camera2: Camera, camera1_positions, camera2_positions):
    assert camera1.extrinsic_matrix is not None, "Camera 1 extrinsic matrix has not been calibrated."
    assert camera2.extrinsic_matrix is not None, "Camera 2 extrinsic matrix has not been calibrated."
    camera1_undistorted = camera1.undistort(camera1_positions)
    camera2_undistorted = camera2.undistort(camera2_positions)
    # `undistort` undoes the intrinsic matrix
    # therefore here, you must pretend that the intrinsic matrix is the identity
    # the *_extrinsic_matrix parameters are supposed to be camera matrices (i.e. intrinsic @ extrinsic)
    # but we just use extrinsic to pretend that the intrinsic matrix is the identity
    # Shape: [4, n]
    # i.e., [{x, y, z, homogenous}, n]
    triangulated_homogenous = cv2.triangulatePoints(camera1.extrinsic_matrix, camera2.extrinsic_matrix, camera1_undistorted, camera2_undistorted)
    triangulated_homogenous = triangulated_homogenous.T
    # Divide xyz by homogenous points
    triangulated = triangulated_homogenous[:, :3] / triangulated_homogenous[:, -1]
    return triangulated

def get_cameras():
    k4a_devices = [pyk4a.PyK4A(device_id=i) for i in [0, 1]]
    k4a_device_map = {}
    for device in k4a_devices:
        device.start()
        k4a_device_map[device.serial] = device

    left = Camera(k4a_device_map['000256121012'])
    right = Camera(k4a_device_map['000243521012'])
    return left, right
