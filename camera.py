import cv2
import numpy as np
import pyk4a
from typing import Optional

class Camera:
    def __init__(self, k4a: pyk4a.PyK4A):
        self.k4a = k4a
        # Camera parameters
        self.rvec_tvec = None
        self.extrinsic_matrix = None
        self.intrinsic_matrix = k4a.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
        self.distortion_coefficients = k4a.calibration.get_distortion_coefficients(pyk4a.CalibrationType.COLOR)
        self.prev_capture: Optional[pyk4a.PyK4ACapture] = None

    def capture(self):
        self.prev_capture = self.k4a.get_capture()
        return self.prev_capture

    def infer_extrinsics(self, apriltag_points, apriltag_object_points):
        ret, rvec, tvec = cv2.solvePnP(apriltag_object_points, apriltag_points, self.intrinsic_matrix, self.distortion_coefficients)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translatoin = tvec
        extrinsics = np.concatenate((rotation_matrix, translatoin), axis=1)

        self.rvec_tvec = (rvec, tvec)
        self.extrinsic_matrix = extrinsics

    def project_points(self, object_points):
        assert self.extrinsic_matrix is not None
        
        rvec, tvec = self.rvec_tvec
        points, _jacobian = cv2.projectPoints(object_points, rvec, tvec, self.intrinsic_matrix, self.distance_coefficients)
        points = points[:, 0, :]

        return points
    
    def undistort(self, image_points):
        # Note: This undoes the intrinsic matrix as well
        return cv2.undistortPoints(image_points.astype(np.float32), self.intrinsic_matrix, self.distortion_coefficients)
    
    def close(self):
        self.k4a.close()

def triangulate(camera1: Camera, camera2: Camera, camera1_positions, camera2_positions):
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
